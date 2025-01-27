# imports
import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
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
                 llm_frozen: str = "True",
                 gnn_model_name: str = "gt",
                 gnn_in_dim: int = 1024, # corresponds to rwr output dimension
                 gnn_hidden_dim: int = 1024,
                 gnn_num_layers: int = 4,
                 gnn_dropout: float = 0.0,
                 gnn_num_heads: int = 4,
                 **kwargs):
        
        super().__init__()
        self.max_txt_len = max_txt_len
        self.max_new_tokens = max_new_tokens

        print('Loading LLaMA...')
        kwargs = {
            "device_map": "auto",
            "revision": "main"
        }

        # create tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path, 
                                                       use_fast=False, 
                                                       revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        # create model
        model = AutoModelForCausalLM.from_pretrained(llm_model_path,
                                                     torch_dtype=torch.float16,
                                                     low_cpu_mem_usage=True,
                                                     **kwargs)
        
        # freeze model if specified
        if llm_frozen == "True":
            print("Freezing LLaMA!")
            for name, param in model.named_parameters():
                param.requires_grad = False # don't update weights for these layers
        else:
            print("Training LLaMA with LORA!")
            model = prepare_model_for_kbit_training(model)

            # lora params
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
    
        self.model = model
        print("Finished loading LLaMA!")

        # load graph encoder component - load_gnn_model function is from GraphEncoder.py 
        self.graph_encoder = load_gnn_model[gnn_model_name](
            in_channels = gnn_in_dim,
            out_channels = gnn_hidden_dim,
            hidden_channels = gnn_hidden_dim,
            num_layers = gnn_num_layers,
            dropout = gnn_dropout,
            num_heads=gnn_num_heads
        ).to(self.model.device)

        # add MLP to project graph encoder output to match llm input
        self.projector = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096)
        ).to(self.model.device)

        # get word embeddings - where is this coming from??
        self.word_embedding = self.model.model.get_input_embeddings()

    # property decorator allows us to call the method without any parameters, and you can assign values to it
    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.bfloat16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.amp.autocast('cuda', dtype=dtype)
        else:
            return contextlib.nullcontext() # return a context manager if autocast off
    
    def encode_graphs(self, samples):
        # use graph encoder module to "compress" graph dimensions
        graphs = samples['graph']
        graphs = graphs.to(self.model.device)
        n_embeds = self.graph_encoder(graphs.x, graphs.edge_index.long())

        # mean pooling: reduce objects along given dimension with reduction operation
        g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')

        return g_embeds
    
    # forward pass (training)
    def forward(self, samples):

        # encode description, questions, and labels
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)

        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples) # encode_graphs only looks at graph description stuff
        graph_embeds = self.projector(graph_embeds)

        # batch processing
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []

        # go batch-by-batch
        for i in range(batch_size):

            # add bos, eos token
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids

            # combine the label input ids with the desc, questions, and special tokens to make full string
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids 

            # embed input
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0]) # attend to all tokens

            # mask out the input graph, description, question in our labels
            label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids 
            batch_label_input_ids.append(label_input_ids)

        # pad input_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i] # mask everything that's not part of the answer - we only want the model to output this 
        
        # convert lists into tensors
        inputs_embeds = torch.stack(batch_inputs_embeds, dim = 0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        # send inp, mask, and labels to the model
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=label_input_ids
            )

        # they only return the loss, hmmm....
        return outputs.loss
    
    # forward pass (inference)
    def inference(self, samples):

        # encode descriptions and questions
        questions = self.tokenizer(samples["question"], add_special_tokens=False)
        descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

        # encode special tokens
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

        # encode graphs
        graph_embeds = self.encode_graphs(samples)
        graph_embeds = self.projector(graph_embeds)

        # batch processing
        batch_size = len(samples['id'])
        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            # add bos, eos
            input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

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
                use_cache=True # IMPORTANT (apparently, not sure why)
            )
        
        # convert tokens back to text
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # return dict of pred + other info
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
        
        return f"Trainable parameters: {trainable_params} | Total parameters: {all_param} | {(trainable_params / all_param):.2%} trainable "






        
