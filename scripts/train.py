# imports 
import time
import os, sys
import wandb
import tqdm.notebook as tqdm
import transformers
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
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
## CONFIG
seed = 42
T = 512
B = 4

# args from config.py
args = parse_args_llama()

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

# ---------------------------------------------------------
## SET UP DDP / DATALOADERS
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) # each gpu has a different 'rank'
    ddp_local_rank = int(os.environ['LOCAL_RANK']) # only used in a multi-node setting
    ddp_world_size = int(os.environ['WORLD_SIZE']) # num. processes running (8 for us)
    device = f'cuda:{ddp_local_rank}' # specify device with local rank
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # process 0 does logging, checkpointing, etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True

    # attempt to auto-detect device
    device = "cpu"
    if torch.cuda.is_available():
        device = GetLowestGPU()
    elif hasattr(torch.backends, "mps"):
        device = "mps"
    print(f"using device: {device}")

seed_everything(seed)

# make dataloaders
train_loader = DataLoader(train_dataset, 
                          batch_size=B,
                          drop_last=True,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=16)

val_loader = DataLoader(val_dataset, 
                          batch_size=B,
                          drop_last=True,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=16)

test_loader = DataLoader(test_dataset, 
                          batch_size=B,
                          drop_last=True,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn,
                          num_workers=16)

# ---------------------------------------------------------
## TRAINING

# change precision
torch.set_float32_matmul_precision('high') # instead of 'highest'

model = GraphLLM(max_text_len=T,
                 max_max_new_tokens=32,
                 max_memory=[80, 80],
                 llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                 llm_frozen='True',
                 revision="main") # args are defaulted in the class

# use_compile option to compile beforehand
use_compile = False
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # unwrapped model

# options
num_training_steps = args.num_epochs * len(train_loader)
progress_bar = tqdm.tqdm(range(num_training_steps))
best_val_loss = float('inf')
log_path = '../logs/graph_llm_no_text/'
save_path = '../checkpoints/graph_llm_no_text/'

# set optimizer
params = [p for _, p in model.named_parameters() if p.requires_grad] # only update non-frozen params (graph encoder)
optimizer = torch.optim.AdamW(
    [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
    betas=(0.9, 0.95)
)

## TRAIN LOOP
for epoch in range(args.num_epochs):

    model.train()
    epoch_loss, accum_loss = 0., 0.

    for step, batch in enumerate(train_loader):

        optimizer.zero_grad()
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            loss = model(batch)
        if ddp:
            model.require_backward_grad_sync = ((step + 1) % args.grad_steps == 0)
        loss.backward()

        if ddp:
            dist.all_reduce(loss, op = dist.ReduceOp.AVG)

        # clip gradients so large changes don't occur - super small clipping too
        clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
        
        # grad steps is a hyprparameter
        if (step + 1) % args.grad_steps == 0:
            adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
    
        epoch_loss += loss.item()
        accum_loss += loss.item()

        if ddp:
            dist.all_reduce(accum_loss, op = dist.ReduceOp.AVG)
            dist.all_reduce(epoch_loss, op=dist.ReduceOp.AVG)

        optimizer.step() # update weights

        if (step + 1) % args.grad_steps == 0:
            lr = optimizer.param_groups[0]['lr']
            # wandb.log({'Lr': lr})
            # wandb.log({'Train Loss': accum_loss / args.grad_steps})
            accum_loss = 0.0
        
        progress_bar.update(1)
    

    # validation
    val_loss = 0.
    eval_output = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            loss = model(batch)
            val_loss += loss.item()
        val_loss /= len(val_loader)
    
    if ddp:
        dist.all_reduce(val_loss, op=dist.ReduceOp.AVG)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        _save_checkpoint(model, optimizer, epoch, args, save_path, is_best=True)
        best_epoch = epoch
    
    torch.cuda.synchronize()
    
    if master_process:
        print(f"Epoch {epoch}/{args.num_epochs} | Train Loss (Epoch Mean): {epoch_loss / len(train_loader)} | Validation Loss: {val_loss} | Best Validation Loss: {best_val_loss} at epoch {best_epoch}", end="\r")

        if epoch - best_epoch >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
if ddp:
    destroy_process_group()

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

# ---------------------------------------------------------
# EVALUATION

# test inference on one batch
batch = next(iter(test_loader))
out = model.inference(batch)
print(out)