# imports 
import os
import time
import sys
import wandb
import tqdm.auto as tqdm
import transformers
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
from torch.distributed.fsdp import MixedPrecision, FullyShardedDataParallel as FSDP
from importlib import reload
from torch_scatter import scatter
from transformers import pipeline
from DGXutils import GetLowestGPU
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, DeepSpeedPlugin

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

# ----------------
## SETUP FUNCTIONS
# ----------------
def main():
    # -------
    ## CONFIG
    # -------
    args = parse_args_llama()
    seed_everything(42)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_steps,
    )
    accelerator.print(f"Initialized accelerator with FSDP")

    # ------------
    ## DATALOADERS
    # ------------
    data_path = '../data/subgraphs/all'
    dataset = BiologicalDataset(data_path)
    idx_split = dataset.get_idx_split()

    # split datasets on idx
    train_dataset = Subset(dataset, idx_split['train'])
    val_dataset = Subset(dataset, idx_split['val'])

    # make dataloaders
    B = 1
    train_loader = DataLoader(train_dataset, 
                            batch_size=B,
                            collate_fn=collate_fn,
                            shuffle=True)

    val_loader = DataLoader(val_dataset, 
                            batch_size=B,
                            collate_fn=collate_fn,
                            shuffle=False)

    # -----------
    ## MODEL INIT
    # -----------
    T = 256
    model = GraphLLM(max_txt_len=T,
                    max_new_tokens=128,
                    llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                    llm_frozen=False, # set frozen to false so we can train with RL
                    fsdp=True, 
                    ) # args are defaulted in the class

    # --------------------
    ## OPTIMIZER & OPTIONS
    # --------------------
    # set optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad] # only update non-frozen params (graph encoder)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # enable grad checkpointing for additional mem savings
    if hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
        accelerator.print("Enabled gradient checkpointing for HF model.")

    # options
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm.tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    best_val_loss = float('inf')
    best_epoch = 0
    save_path = '../checkpoints/graph_llm_fsdp/'
    log_path = '../logs/graph_llm_fsdp/'

    ## TRAIN LOOP
    if accelerator.is_main_process:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log = os.path.join(log_path, 'log.txt')
        with open(log, 'w') as f:
            f.write("epoch,train_loss,val_loss\n")
    
    iter_num = 0
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        # backprop
        for step, batch in enumerate(train_loader):
            # grad accumulation
            with accelerator.accumulate(model):
                loss = model(batch)
                accelerator.backward(loss)
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            iter_num += 1
            progress_bar.update(1)
            adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
        train_loss = epoch_loss / len(train_loader)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss = model(batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        # print epoch stats
        accelerator.print(f"Epoch {epoch}/{args.num_epochs} | "
                    f"Train Loss: {epoch_loss / len(train_loader):.4f} | "
                    f"Validation Loss: {val_loss:.4f} | "
                    f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}", 
                    end="\r")
        
        # save checkpoint if best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                _save_checkpoint(unwrapped_model, optimizer, epoch, args, save_path, is_best=True)
        accelerator.wait_for_everyone()

        # checkpoint and save to log
        if accelerator.is_main_process:
            with open(log, 'a') as f:
                f.write(f"{epoch},{iter_num},{train_loss},{val_loss}\n")

        # clear cached mem
        torch.cuda.clear_cache()

        # Early stopping if needed
        if epoch - best_epoch >= args.patience:
            accelerator.print(f"\nEarly stopping at epoch {epoch}")
            break
    accelerator.print("Training Complete")

if __name__ == '__main__':
    main()