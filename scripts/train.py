import os, sys
import wandb
import tqdm.notebook as tqdm
import transformers
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from importlib import reload
from torch_scatter import scatter
from transformers import pipeline

sys.path.append('../')

from utils import preprocess as pp
from utils.graph_llm import GraphLLM
from utils.llm import LLM
from utils.multiplex import Multiplex
from utils.textualize import *
from utils.bio_graphs import BiologicalDataset

# training imports
from utils.evaluate import eval_funcs
from utils.config import parse_args_llama
from utils.ckpt import _save_checkpoint, _reload_best_model
from utils.collate import collate_fn
from utils.seed import seed_everything
from utils.lr_schedule import adjust_learning_rate

# ----------------------------------------------------------
# OPTIONS
batch_size = 1
seed = 42
args = parse_args_llama()
seed_everything(seed)

# ----------------------------------------------------------
# DATASET
data_path = '../data/DREAM4_gold_standards/connections_node_label'
dataset = BiologicalDataset(data_path)
idx_split = dataset.get_idx_split()

# split datasets on idx
train_dataset = [dataset[i] for i in idx_split["train"]]
val_dataset = [dataset[i] for i in idx_split["val"]]
test_dataset = [dataset[i] for i in idx_split["test"]]

# make dataloaders
train_loader = DataLoader(train_dataset, 
                          batch_size=batch_size,
                          drop_last=True,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn)

val_loader = DataLoader(val_dataset, 
                          batch_size=batch_size,
                          drop_last=True,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn)

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          drop_last=True,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn)

# ----------------------------------------------------------
# LOAD MODEL
model = GraphLLM(max_text_len=512,
                     max_max_new_tokens=32,
                     max_memory=[80, 80],
                     llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                     llm_frozen='True',
                     revision="main") # args are defaulted in the class

# ----------------------------------------------------------
# TRAINING
num_training_steps = args.num_epochs * len(train_loader)
progress_bar = tqdm.tqdm(range(num_training_steps))
best_val_loss = float('inf')
save_path = '../checkpoints/test_run/'

# wandb.init(project=f"{args.project}",
#             name=f"{dataset}_{args.model_name}_seed{seed}",
#             config=args)

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
        loss = model(batch)
        
        # clip gradients so large changes don't occur - super small clipping too
        clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
        
        # grad steps is a hyprparameter
        if (step + 1) % args.grad_steps == 0:
            adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
        
        optimizer.step()
        epoch_loss += loss.item()
        accum_loss += loss.item()

        if  (step + 1) % args.grad_steps == 0:
            lr = optimizer.param_groups[0]['lr']
            # wandb.log({'Lr': lr})
            # wandb.log({'Train Loss': accum_loss / args.grad_steps})
            accum_loss = 0.
        
        progress_bar.update(1)
    
    print(f"Epoch {epoch}/{args.num_epochs} | Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")
    # wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})

    # validation
    val_loss = 0.
    eval_output = []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            loss = model(batch)
            val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch}/{args.num_epochs} | Validation Loss: {val_loss}")
        # wandb.log({'Validation Loss': val_loss})
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        _save_checkpoint(model, optimizer, epoch, args, save_path, is_best=True)
        best_epoch = epoch
    
    print(f"Epoch {epoch}/{args.num_epochs} | Best Validation Loss: {best_val_loss} at epoch {best_epoch}")

    if epoch - best_epoch >= args.patience:
        print(f"Early stopping at epoch {epoch}")
        break

torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

print("Training complete!")