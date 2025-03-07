# imports
import contextlib
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from .graph_encoder import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

# define special tokens
BOS = '<|begin_of_text|>'
EOS_USER = '<|endoftext|>'
EOS = '<|endoftext|>'

# define mask token
IGNORE_INDEX = -100

class GraphLLM(nn.Module):
    """
    Graph LLM object, re-implemented from G-Retriever: https://github.com/XiaoxinHe/G-Retriever/blob/main/src/model/graph_llm.py
    """
    def __init__(self,
                 max_txt_len: int = 512,
                 max_new_tokens: int = 32,
                 llm_model_path: str = "meta-llama/Meta-Llama-3-8B-Instruct",
                 llm_frozen: bool = True,
                 gnn_model_name: str = "gt",
                 gnn_in_dim: int = 1024, # corresponds to rwr output dimension
                 gnn_hidden_dim: int = 1024,
                 gnn_num_layers: int = 4,
                 gnn_dropout: float = 0.0,
                 gnn_num_heads: int = 4,
                 fsdp: bool = False,
                 **kwargs):
        
        super().__init__()
        self.max_txt_len = max_txt_len
        self.max_new_tokens = max_new_tokens
        self.fsdp = fsdp
        self.gnn_model_name = gnn_model_name

        if self.fsdp:
            # set up tokenizer and llm
            kwargs = {
                "revision": "main",
            }
        else:
            # set up tokenizer and llm
            kwargs = {
                "revision": "main",
                "device_map": "auto"
            }

        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, 
                                                       use_fast=False, 
                                                       revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        # create model
        model = AutoModelForCausalLM.from_pretrained(llm_model_path,
                                                     torch_dtype=torch.float32,
                                                     low_cpu_mem_usage=True,
                                                     **kwargs)
        
        # freeze model if specified
        if llm_frozen:
            for name, param in model.named_parameters():
                param.requires_grad = False # don't update weights for these layers
            print("Model is frozen")
        else:
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = ["q_proj", "v_proj"]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules = lora_target_modules,
                lora_dropout = lora_dropout,
                bias="none",
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, config)
            print("Training with Lora")
        self.model = model

        # store gnn + proj as submodules so they're not touched by accelerate
        graph_encoder = load_gnn_model[gnn_model_name](
            in_channels=gnn_in_dim,
            out_channels=gnn_hidden_dim,
            hidden_channels=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
            num_heads=gnn_num_heads
        ).float().to(self.model.device)

        self.projector = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096)
        ).to(self.model.device)

        # store graph_encoder in this dictionary so it's not touched by accelerate
        self._fp32_modules = {
            "graph_encoder": graph_encoder,
        }
            
        # get word embeddings - where is this coming from??
        self.word_embedding = self.model.model.get_input_embeddings()

        # Register constant embeddings (computed once) as buffers.
        with torch.no_grad():
            bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
            self.register_buffer("bos_embeds", bos_embeds)

            pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)
            self.register_buffer("pad_embeds", pad_embeds)

            self.eos_tokens = self.tokenizer(EOS, add_special_tokens=False).input_ids
            self.eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False).input_ids

    # property decorator allows us to call the method without any parameters, and you can assign values to it
    @property
    def device(self):
        return next(self.parameters()).device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.bfloat16
        if self.device != torch.device("cpu"):
            return torch.amp.autocast('cuda', dtype=dtype)
        else:
            return contextlib.nullcontext() # return a context manager if autocast off
    
    def encode_graphs(self, samples):
        graphs = samples['graph'].to(self.model.device)
        graph_encoder = self._fp32_modules["graph_encoder"]

        # keep ops in fp32, not half
        with torch.amp.autocast('cuda', enabled=False):
            if self.gnn_model_name == "gat":
                n_embeds, _ = graph_encoder(graphs.x, graphs.edge_index.long(), graphs.edge_attr) 
            else:
                n_embeds = graph_encoder(graphs.x, graphs.edge_index.long())
            g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
        return g_embeds.bfloat16() if self.fsdp else g_embeds # turn back to half so it works with everything else

    def forward(self, samples):
        device = self.model.device  # cache device

        # Encode graphs and project them
        graph_embeds = self.encode_graphs(samples)  # shape: [batch_size, embed_dim]
        graph_embeds = self.projector(graph_embeds)

        # Tokenize the text inputs
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # Pre-defined special tokens and embeddings
        eos_user_tokens = self.eos_user_tokens
        eos_tokens = self.eos_tokens
        bos_embeds = self.bos_embeds       # shape: [1, embed_dim]
        pad_embeds = self.pad_embeds       # shape: [1, embed_dim]

        batch_size = len(samples['id'])
        # First pass: compute final sequence lengths and cache token IDs.
        lengths = []
        tokenized_ids_list = []
        label_lengths = []
        for i in range(batch_size):
            label_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens
            input_ids = (descriptions.input_ids[i][:self.max_txt_len] +
                         questions.input_ids[i] +
                         eos_user_tokens +
                         label_ids)
            # Each sample is: [bos] + [graph] + token_embeds(input_ids)
            sample_length = bos_embeds.size(0) + 1 + len(input_ids)
            lengths.append(sample_length)
            label_lengths.append(len(label_ids))
            tokenized_ids_list.append(input_ids)

        max_length = max(lengths)
        embed_dim = bos_embeds.size(1)

        # Preallocate batched tensors (using left-padding)
        # All sequences will be left-padded with pad_embeds.
        inputs_embeds = pad_embeds.expand(max_length, -1).clone().repeat(batch_size, 1, 1)
        attention_mask = torch.zeros((batch_size, max_length), device=device, dtype=torch.long)
        label_input_ids = torch.full((batch_size, max_length), IGNORE_INDEX, device=device, dtype=torch.long)

        # Second pass: fill in each row at the correct position (right-aligned)
        for i in range(batch_size):
            cur_len = lengths[i]
            pad_len = max_length - cur_len

            # Build sample embeddings: [bos] + [graph_embed] + [token_embeds]
            main_ids_tensor = torch.tensor(tokenized_ids_list[i], device=device, dtype=torch.long)
            token_embeds = self.word_embedding(main_ids_tensor)
            sample_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), token_embeds], dim=0)
            inputs_embeds[i, pad_len:] = sample_embeds

            # Set attention mask (1 for valid tokens)
            attention_mask[i, pad_len:] = 1

            # Build label input IDs: left-pad with IGNORE_INDEX so that the last part holds labels.
            label_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens
            label_start = max_length - len(label_ids)
            label_input_ids[i, label_start:] = torch.tensor(label_ids, device=device, dtype=torch.long)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=False,
                labels=label_input_ids,
            )

        return outputs.loss, outputs

    def inference(self, samples, 
                  num_generations=1):
        # cache device
        device = self.model.device

        # encode description and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        
        # encode special tokens
        eos_user_tokens = self.eos_user_tokens
        bos_embeds = self.bos_embeds
        pad_embeds = self.pad_embeds

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []

        for i in range(batch_size):
            # Add bos & eos token
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length-batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0]*pad_length+batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(device)
        attention_mask = torch.tensor(batch_attention_mask).to(device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                # do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,  # IMPORTANT!
                num_return_sequences=num_generations,
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc'],
                'out_ids': outputs}
    
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return f"Trainable parameters: {trainable_params} | Total parameters: {all_param} | {(trainable_params / all_param):.2%} trainable "