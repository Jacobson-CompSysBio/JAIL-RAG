import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

# special tokens
BOS = '<|begin_of_text|>'
EOS_USER = '<|endoftext|>'
EOS = '<|endoftext|>'

IGNORE_INDEX = -100

# define the model
class LLM(nn.Module):

    def __init__(
            self,
            max_txt_len: int = 512,
            max_new_tokens: int = 32,
            max_memory: list = [80, 80],
            llm_model_path: str = "meta-llama/llama-8b-Instruct",
            llm_frozen: str = "True",
            **kwargs):
        super().__init__()

        # hyperparameters
        self.max_txt_len = max_txt_len
        self.max_new_tokens = max_new_tokens

        print('Loading LLaMA...')
        kwargs ={
            "max_memory": {i: f'{size}GiB' for i, size in enumerate(max_memory)},
            "device_map": "auto",
            "revision": "main"
        }
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path,
                                                       use_fast=False,
                                                       revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        model = AutoModelForCausalLM.from_pretrained(llm_model_path,
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     **kwargs)
        
        if llm_frozen == 'True':
            print('Freezing LLaMA...')
            for name, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLaMA with LORA`...")
            model = prepare_model_for_kbit_training(model)

            # lora params.
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj"
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finished loading LLaMA...')
        self.word_embedding = self.model.model.get_input_embeddings() # get the trained embedding table from the model

    @property
    def device(self):
        return list(self.parameters())[0].device
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, no autocast
        # if on gpu, autocast
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast(dtype=dtype, device_type="cuda")
        else:
            return contextlib.nullcontext()
        
    def forward(self, samples):

        # encode desc, questions, labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # process in batches
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        for i in range(batch_size):

            # add bos and eos tokens
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        
        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])

        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
        
        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        # autocast is for speed - slightly lower mantissa precision
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids
            )
        
        return outputs.loss
    
    def inference(self, samples):

        # encode descriptions and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # batch processing
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(batch_size):

            # add bos and eos tokens
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True, # important
            )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc']}

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return trainable_params, all_param