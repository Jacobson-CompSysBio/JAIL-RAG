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

        # Register constant embeddings (computed once) as buffers.
        with torch.no_grad():
            bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
            self.register_buffer("bos_embeds", self.model.model.get_input_embeddings()(bos_ids))
            # For pad embedding, create a 1-element tensor.
            pad_id = torch.tensor(self.tokenizer.pad_token_id)
            self.register_buffer("pad_embeds", self.model.model.get_input_embeddings()(pad_id).unsqueeze(0))

        # store gnn + proj as submodules so they're not touched by accelerate
        graph_encoder = load_gnn_model[gnn_model_name](
            in_channels=gnn_in_dim,
            out_channels=gnn_hidden_dim,
            hidden_channels=gnn_hidden_dim,
            num_layers=gnn_num_layers,
            dropout=gnn_dropout,
            num_heads=gnn_num_heads
        ).float()
        projector = nn.Sequential(
            nn.Linear(gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096)
        ).float()
        if self.fsdp:
            self._fp32_modules = {
                "graph_encoder": graph_encoder,
                "projector": projector,
            }
        else:
            self.graph_encoder = graph_encoder
            self.projector = projector
        # get word embeddings - where is this coming from??
        self.word_embedding = self.model.model.get_input_embeddings()

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
        x = graphs.x.float()
        # get gnn and projector
        if self.fsdp:
            graph_encoder = self._fp32_modules["graph_encoder"].to(self.device)
            projector = self._fp32_modules["projector"].to(self.device)
        else:
            graph_encoder = self.graph_encoder.to(self.device)
            projector = self.projector.to(self.device)
        # keep ops in fp32, not half
        with torch.amp.autocast('cuda', enabled=False):
            n_embeds = graph_encoder["graph_encoder"](x, graphs.edge_index.long())
            g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
            g_embeds = projector["projector"](g_embeds.float())
        return g_embeds

    def forward(self, samples):

        # encode graphs first to avoid FP16 issues
        graph_embeds = self.encode_graphs(samples)  # Expected shape: [batch_size, embed_dim]

        with self.maybe_autocast(dtype=torch.float16):
            # Tokenize texts for the batch.
            questions = self.tokenizer(samples["question"], add_special_tokens=False)
            descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
            labels = self.tokenizer(samples["label"], add_special_tokens=False)
            eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
            eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)

            batch_size = len(samples['id'])
            batch_inputs_embeds = []
            batch_attention_mask = []
            batch_label_input_ids = []

            for i in range(batch_size):
                # Prepare label token IDs (truncate and append EOS)
                label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
                # Construct the full input IDs by concatenating the parts.
                input_ids = (descriptions.input_ids[i][:self.max_txt_len] +
                            questions.input_ids[i] +
                            eos_user_tokens.input_ids +
                            label_input_ids)
                # Create tensor for token IDs.
                input_ids_tensor = torch.tensor(input_ids, device=self.model.device, dtype=torch.long)
                # Obtain embeddings from the LLM's embedding layer.
                inputs_embeds = self.word_embedding(input_ids_tensor)
                # Prepend the constant BOS embedding and the graph embedding.
                sample_embeds = torch.cat([self.bos_embeds,
                                        graph_embeds[i].unsqueeze(0),
                                        inputs_embeds], dim=0)
                batch_inputs_embeds.append(sample_embeds)
                # Create attention mask for this sample.
                attn_mask = torch.ones(sample_embeds.shape[0], device=self.model.device, dtype=torch.long)
                batch_attention_mask.append(attn_mask)
                # Build target labels: mask out tokens before the answer.
                num_mask = sample_embeds.shape[0] - len(label_input_ids)
                masked_label_ids = torch.tensor([IGNORE_INDEX] * num_mask + label_input_ids,
                                                device=self.model.device, dtype=torch.long)
                batch_label_input_ids.append(masked_label_ids)

            # Pad the batch of sequences.
            inputs_embeds = torch.nn.utils.rnn.pad_sequence(batch_inputs_embeds, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(batch_attention_mask, batch_first=True, padding_value=0)
            label_input_ids = torch.nn.utils.rnn.pad_sequence(batch_label_input_ids, batch_first=True, padding_value=IGNORE_INDEX)

            # Forward pass through the LLM.
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=label_input_ids
            )
        return outputs.loss
    
    def inference(self, samples):
        with self.maybe_autocast(dtype=torch.float16):
            questions = self.tokenizer(samples["question"], add_special_tokens=False)
            descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
            eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
            
            batch_size = len(samples['id'])
            batch_inputs_embeds = []
            batch_attention_mask = []
            # Encode graph.
            graph_embeds = self.encode_graphs(samples)
            
            for i in range(batch_size):
                input_ids = (
                    descriptions.input_ids[i][:self.max_txt_len] +
                    questions.input_ids[i] +
                    eos_user_tokens.input_ids
                )
                input_ids_tensor = torch.tensor(input_ids, device=self.model.device)
                inputs_embeds = self.word_embedding(input_ids_tensor)
                sample_embeds = torch.cat([self.bos_embeds.unsqueeze(0),
                                           graph_embeds[i].unsqueeze(0),
                                           inputs_embeds], dim=0)
                batch_inputs_embeds.append(sample_embeds)
                batch_attention_mask.append([1] * sample_embeds.shape[0])
            
            inputs_embeds = torch.nn.utils.rnn.pad_sequence(batch_inputs_embeds, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x, device=self.model.device) for x in batch_attention_mask],
                batch_first=True,
                padding_value=0
            )
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=True
            )
            
            pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return {
                'id': samples['id'],
                'pred': pred,
                'label': samples['label'],
                'question': samples['question'],
                'desc': samples['desc']
            }
    
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return f"Trainable parameters: {trainable_params} | Total parameters: {all_param} | {(trainable_params / all_param):.2%} trainable "
