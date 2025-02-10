# imports
import contextlib
import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
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
                 fsdp: bool = False,
                 **kwargs):
        
        super().__init__()
        self.max_txt_len = max_txt_len
        self.max_new_tokens = max_new_tokens

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
        if llm_frozen == "True":
            for name, param in model.named_parameters():
                param.requires_grad = False # don't update weights for these layers
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
    
        self.model = model

        # Register constant embeddings (computed once) as buffers.
        with torch.no_grad():
            bos_ids = self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0]
            self.register_buffer("bos_embeds", self.model.model.get_input_embeddings()(bos_ids))
            # For pad embedding, create a 1-element tensor.
            pad_id = torch.tensor(self.tokenizer.pad_token_id)
            self.register_buffer("pad_embeds", self.model.model.get_input_embeddings()(pad_id).unsqueeze(0))

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

        # half precision
        self.graph_encoder = self.graph_encoder.half()
        self.projector = self.projector.half()

        # if using fsdp, wrap only trainable modules
        if fsdp:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

            # wrap only ge + proj modules
            self.graph_encoder = FSDP(self.graph_encoder,
                                      mixed_precision=mixed_precision_policy,
                                      use_orig_params=False)
            self.projector = FSDP(self.projector,
                                  mixed_precision=mixed_precision_policy,
                                  use_orig_params=False)

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
    
    # def encode_graphs(self, samples):
    #     # use graph encoder module to "compress" graph dimensions
    #     graphs = samples['graph']
    #     graphs = graphs.to(self.model.device)
    #     n_embeds = self.graph_encoder(graphs.x, graphs.edge_index.long())

    #     # mean pooling: reduce objects along given dimension with reduction operation
    #     g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')

    #     return g_embeds
    
    def encode_graphs(self, samples):
        # Assume samples['graph'] is a graph data object.
        graphs = samples['graph'].to(self.model.device)
        with self.maybe_autocast(dtype=torch.float16):
            n_embeds = self.graph_encoder(graphs.x, graphs.edge_index.long())
            # Use mean pooling.
            g_embeds = scatter(n_embeds, graphs.batch, dim=0, reduce='mean')
            g_embeds = self.projector(g_embeds)
        return g_embeds
    
    # # forward pass (training)
    # def forward(self, samples):

    #     # encode description, questions, and labels
    #     questions = self.tokenizer(samples["question"], add_special_tokens=False)
    #     descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
    #     labels = self.tokenizer(samples["label"], add_special_tokens=False)

    #     # encode special tokens
    #     eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
    #     eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
    #     bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
    #     pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

    #     # encode graphs
    #     graph_embeds = self.encode_graphs(samples) # encode_graphs only looks at graph description stuff
    #     graph_embeds = self.projector(graph_embeds)

    #     # batch processing
    #     batch_size = len(samples['id'])
    #     batch_inputs_embeds = []
    #     batch_attention_mask = []
    #     batch_label_input_ids = []

    #     # go batch-by-batch
    #     for i in range(batch_size):

    #         # add bos, eos token
    #         label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids

    #         # combine the label input ids with the desc, questions, and special tokens to make full string
    #         input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids + label_input_ids 

    #         # embed input
    #         inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
    #         inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

    #         batch_inputs_embeds.append(inputs_embeds)
    #         batch_attention_mask.append([1] * inputs_embeds.shape[0]) # attend to all tokens

    #         # mask out the input graph, description, question in our labels
    #         label_input_ids = [IGNORE_INDEX] * (inputs_embeds.shape[0] - len(label_input_ids)) + label_input_ids 
    #         batch_label_input_ids.append(label_input_ids)

    #     # pad input_embeds
    #     max_length = max([x.shape[0] for x in batch_inputs_embeds])
    #     for i in range(batch_size):
    #         pad_length = max_length - batch_inputs_embeds[i].shape[0]
    #         batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
    #         batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
    #         batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length+batch_label_input_ids[i] # mask everything that's not part of the answer - we only want the model to output this 
        
    #     # convert lists into tensors
    #     inputs_embeds = torch.stack(batch_inputs_embeds, dim = 0).to(self.model.device)
    #     attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
    #     label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

    #     # send inp, mask, and labels to the model
    #     with self.maybe_autocast():
    #         outputs = self.model(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             labels=label_input_ids
    #         )

    #     # they only return the loss, hmmm....
    #     return outputs.loss

    def forward(self, samples):
        # Wrap most of the computation in AMP autocast.
        with self.maybe_autocast(dtype=torch.float16):
            # Tokenize texts.
            questions = self.tokenizer(samples["question"], add_special_tokens=False)
            descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)
            labels = self.tokenizer(samples["label"], add_special_tokens=False)

            # Tokenize special tokens.
            eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
            eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)

            # Encode graph.
            graph_embeds = self.encode_graphs(samples)

            batch_size = len(samples['id'])
            batch_inputs_embeds = []
            batch_attention_mask = []
            batch_label_input_ids = []

            # Process each sample in the batch.
            for i in range(batch_size):
                # Construct label input ids (truncate if needed, then append EOS token).
                label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
                # Construct input ids: description (truncated) + question + special tokens + label.
                input_ids = (
                    descriptions.input_ids[i][:self.max_txt_len] +
                    questions.input_ids[i] +
                    eos_user_tokens.input_ids +
                    label_input_ids
                )
                # Convert to tensor on the correct device.
                input_ids_tensor = torch.tensor(input_ids, device=self.model.device)
                inputs_embeds = self.word_embedding(input_ids_tensor)
                # Prepend the cached BOS embedding and the corresponding graph embedding.
                # Unsqueeze graph embedding to add a sequence dimension.
                sample_embeds = torch.cat([self.bos_embeds.unsqueeze(0),
                                           graph_embeds[i].unsqueeze(0),
                                           inputs_embeds], dim=0)
                batch_inputs_embeds.append(sample_embeds)
                batch_attention_mask.append([1] * sample_embeds.shape[0])
                # Mask out non-target tokens: only the label tokens are used for loss.
                masked_label_ids = [IGNORE_INDEX] * (sample_embeds.shape[0] - len(label_input_ids)) + label_input_ids
                batch_label_input_ids.append(masked_label_ids)

            # Use torch.nn.utils.rnn.pad_sequence to pad the sequences.
            inputs_embeds = torch.nn.utils.rnn.pad_sequence(batch_inputs_embeds, batch_first=True, padding_value=0)
            attention_mask = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x, device=self.model.device) for x in batch_attention_mask],
                batch_first=True,
                padding_value=0
            )
            label_input_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x, device=self.model.device) for x in batch_label_input_ids],
                batch_first=True,
                padding_value=IGNORE_INDEX
            )
            
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
    
    # # forward pass (inference)
    # def inference(self, samples):

    #     # encode descriptions and questions
    #     questions = self.tokenizer(samples["question"], add_special_tokens=False)
    #     descriptions = self.tokenizer(samples["desc"], add_special_tokens=False)

    #     # encode special tokens
    #     eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
    #     bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0])
    #     pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id)).unsqueeze(0)

    #     # encode graphs
    #     graph_embeds = self.encode_graphs(samples)
    #     graph_embeds = self.projector(graph_embeds)

    #     # batch processing
    #     batch_size = len(samples['id'])
    #     batch_inputs_embeds = []
    #     batch_attention_mask = []
    #     for i in range(batch_size):
    #         # add bos, eos
    #         input_ids = descriptions.input_ids[i][:self.max_txt_len] + questions.input_ids[i] + eos_user_tokens.input_ids
    #         inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
    #         inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)

    #         batch_inputs_embeds.append(inputs_embeds)
    #         batch_attention_mask.append([1] * inputs_embeds.shape[0])
        
    #     # pad inputs_embeds
    #     max_length = max([x.shape[0] for x in batch_inputs_embeds])
    #     for i in range(batch_size):
    #         pad_length = max_length - batch_inputs_embeds[i].shape[0]
    #         batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
    #         batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

    #     inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
    #     attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

    #     with self.maybe_autocast():
    #         outputs = self.model.generate(
    #             inputs_embeds=inputs_embeds,
    #             max_new_tokens=self.max_new_tokens,
    #             attention_mask=attention_mask,
    #             use_cache=True # IMPORTANT (apparently, not sure why)
    #         )
        
    #     # convert tokens back to text
    #     pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    #     # return dict of pred + other info
    #     return {'id': samples['id'],
    #             'pred': pred,
    #             'label': samples['label'],
    #             'question': samples['question'],
    #             'desc': samples['desc']}
    
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return f"Trainable parameters: {trainable_params} | Total parameters: {all_param} | {(trainable_params / all_param):.2%} trainable "






        
