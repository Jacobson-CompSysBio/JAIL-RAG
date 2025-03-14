# imports 
import os

# special cuda stuff
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'

import gc
import time
import sys
import wandb
import tqdm.auto as tqdm
import transformers
from datetime import timedelta
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
from accelerate.utils import DistributedDataParallelKwargs, DeepSpeedPlugin, InitProcessGroupKwargs

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

def main():
    # -------
    ## CONFIG
    # -------
    # set paths
    data_path = '../data/subgraphs/all'
    save_path = '../checkpoints/graph_llm_fsdp/'
    log_path = '../logs/graph_llm_fsdp/'

    # other config
    args = parse_args_llama()
    seed_everything(42)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    # os.environ['NCCL_DEBUG'] = 'INFO'
    
    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=86400))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_steps,
        kwargs_handlers=[kwargs]
    )
    accelerator.print(f"Initialized accelerator with FSDP")

    # ------------
    ## DATALOADERS
    # ------------
    dataset = BiologicalDataset(data_path)
    idx_split = dataset.get_idx_split()

    # split datasets on idx
    train_dataset = Subset(dataset, idx_split['train'])
    val_dataset = Subset(dataset, idx_split['val'])

    # make dataloaders
    B = 8
    train_loader = DataLoader(train_dataset, 
                            batch_size=B,
                            collate_fn=collate_fn,
                            shuffle=False)

    val_loader = DataLoader(val_dataset, 
                            batch_size=B,
                            collate_fn=collate_fn,
                            shuffle=False)

    # -----------
    ## MODEL INIT
    # -----------
    T = 128
    model = GraphLLM(max_txt_len=T,
                    max_new_tokens=32,
                    llm_model_path='meta-llama/Meta-Llama-3.1-8B-Instruct',
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

    for name, module in model._fp32_modules.items():
        model._fp32_modules[name] = module.to(accelerator.device)

    # enable grad checkpointing for additional mem savings
    if hasattr(model.model, "gradient_checkpointing_enable"):
        model.model.gradient_checkpointing_enable()
        accelerator.print("Enabled gradient checkpointing for HF model.")

    # options
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm.tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    best_val_loss = float('inf')
    best_epoch = 0

    ## TRAIN LOOP
    if accelerator.is_main_process:
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log = os.path.join(log_path, 'log.txt')
        with open(log, 'w') as f:
            f.write("epoch,iter,train_loss,val_loss\n")
    
    iter_num = 0
    for epoch in range(args.num_epochs):
        accelerator.print(f"Epoch {epoch}/{args.num_epochs}")
        accelerator.print("Setting Sampler...")
        if hasattr(train_loader, 'sampler') and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0

        accelerator.print("Backprop...")

        # synchronize gpus
        accelerator.wait_for_everyone()
        torch.cuda.synchronize()

        # backprop
        for step, batch in enumerate(train_loader):
            loss = model(batch)
            accelerator.backward(loss)

            # sync again to avoid out-of-sync buildup
            torch.cuda.synchronize()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # clear memory again
            if step % 50 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            epoch_loss += loss.item()
            iter_num += 1
            progress_bar.update(1)
            adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            # wait for everyone to finish
            if step % 100 == 0:
                accelerator.wait_for_everyone()
                torch.cuda.synchronize()

        train_loss = epoch_loss / len(train_loader)

        accelerator.print("validating...")
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss = model(batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        accelerator.print("Saving checkpoint...")
        # Save checkpoint if we have a new best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # not saving for now, just testing that we can get all the way through
            # accelerator.save_model(model, save_path, safe_serialization=False)
        
        # print epoch stats
        accelerator.print(f"Epoch {epoch}/{args.num_epochs} | "
                    f"Train Loss: {epoch_loss / len(train_loader):.4f} | "
                    f"Validation Loss: {val_loss:.4f} | "
                    f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}")

        # checkpoint and save to log
        if accelerator.is_main_process:
            with open(log, 'a') as f:
                f.write(f"{epoch},{iter_num},{train_loss},{val_loss}\n")
        
        # Early stopping if needed
        if epoch - best_epoch >= args.patience:
            accelerator.print(f"\nEarly stopping at epoch {epoch}...")
            accelerator.end_training()
            break
    
    torch.cuda.empty_cache()
    accelerator.end_training()
    accelerator.print("Training Complete")

if __name__ == '__main__':
    main()