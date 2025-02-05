# imports 
import time
import os, sys
import wandb
import tqdm.auto as tqdm
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from importlib import reload
from torch_scatter import scatter
from transformers import pipeline
from DGXutils import GetLowestGPU

sys.path.append('../')

# custom imports
from utils import preprocess as pp
from utils.graph_llm import GraphLLM
from utils.llm import LLM
from utils.multiplex import Multiplex
from utils.textualize import *
from utils.bio_graphs import BiologicalDataset
from utils.evaluate import eval_funcs
from utils.config import parse_args_llama
from utils.ckpt import _save_checkpoint, _reload_best_model
from utils.collate import collate_fn
from utils.seed import seed_everything
from utils.lr_schedule import adjust_learning_rate
# ---------------------------------------------------------
## SETUP FUNCTIONS

# set up fsdp
def fsdp_setup():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

# clean up fsdp
def fsdp_cleanup():
    dist.destroy_process_group()

# ---------------------------------------------------------
## CONFIG
seed = 42
T = 256
B = 2

# args from config.py
args = parse_args_llama()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# set up fsdp
local_rank = fsdp_setup()

# ---------------------------------------------------------

## LOAD DATA
# get dataset
data_path = '../data/subgraphs/all'
dataset = BiologicalDataset(data_path)
idx_split = dataset.get_idx_split()

# split datasets on idx
train_dataset = Subset(dataset, idx_split['train'])
val_dataset = Subset(dataset, idx_split['val'])
test_dataset = Subset(dataset, idx_split['test'])

# Use DistributedSampler so each GPU gets its share of the data.
train_sampler = DistributedSampler(train_dataset)
val_sampler   = DistributedSampler(val_dataset)
test_sampler  = DistributedSampler(test_dataset)

# ---------------------------------------------------------
## DATALOADERS + FSDP

seed_everything(seed)

# make dataloaders
train_loader = DataLoader(train_dataset, 
                          batch_size=B,
                          sampler=train_sampler,
                          collate_fn=collate_fn)

val_loader = DataLoader(val_dataset, 
                          batch_size=B,
                          sampler=val_sampler,
                          collate_fn=collate_fn)

test_loader = DataLoader(test_dataset, 
                          batch_size=B,
                          sampler=test_sampler,
                          collate_fn=collate_fn)

# ---------------------------------------------------------
## TRAINING

model = GraphLLM(max_text_len=T,
                 max_max_new_tokens=32,
                 max_memory=[80, 80],
                 llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                 llm_frozen='True',
                 revision="main",
                 fsdp=True) # args are defaulted in the class
model.half()

# Set mixed precision policy if desired
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.float16,
    reduce_dtype=torch.float16,
    buffer_dtype=torch.float16,
)

# Move the model to the proper GPU (local_rank)
model.to(local_rank)

# Wrap the model with FSDP
model = FSDP(model, 
             mixed_precision=mixed_precision_policy,
             use_orig_params=True,
             cpu_offload=True)

# options
num_training_steps = args.num_epochs * len(train_loader)
progress_bar = tqdm.tqdm(range(num_training_steps))
best_val_loss = float('inf')
log_path = '../logs/graph_llm_no_text/'
save_path = '../checkpoints/graph_llm_no_text/'

# set optimizer
params = [p for _, p in model.named_parameters() if p.requires_grad] # only update non-frozen params (graph encoder)
optimizer = torch.optim.AdamW(
    [{'params': params, 'lr': args.lr, 'weight_decay': args.wd},],
    betas=(0.9, 0.95)
)

## TRAIN LOOP
for epoch in range(args.num_epochs):

    model.train()
    epoch_loss, accum_loss = 0., 0.
    train_sampler.set_epoch(epoch)

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(local_rank)
        loss = model(batch)
        loss.backward()

        # clip gradients so large changes don't occur - super small clipping too
        clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
        
        # grad steps is a hyprparameter
        if (step + 1) % args.grad_steps == 0:
            adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
    
        epoch_loss += loss.item()
        accum_loss += loss.item()

        optimizer.step() # update weights

        if (step + 1) % args.grad_steps == 0:
            lr = optimizer.param_groups[0]['lr']
            accum_loss = 0.0
        
        progress_bar.update(1)
    
    # validation
    val_loss = 0.
    eval_output = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(local_rank)
            loss = model(batch)
            val_loss += loss.item()
        val_loss /= len(val_loader)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        _save_checkpoint(model, optimizer, epoch, args, save_path, is_best=True)
        best_epoch = epoch
    
    # Print training status (each process prints; you might want to limit this to rank 0)
    if dist.get_rank() == 0:
        print(f"Epoch {epoch}/{args.num_epochs} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
              f"Validation Loss: {val_loss:.4f} | Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}", end="\r")
        
    # Early stopping if needed
    if epoch - best_epoch >= args.patience:
        if dist.get_rank() == 0:
            print(f"\nEarly stopping at epoch {epoch}")
        break

# ---------------------
## CLEANUP
fsdp_cleanup()
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()