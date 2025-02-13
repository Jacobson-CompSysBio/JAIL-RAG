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
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
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

def main():
    # -------
    ## CONFIG
    # -------
    args = parse_args_llama()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(42)

    # accelerate setup
    fsdp_config = {
        "reshard_after_forward": True,
        "min_num_params": 1e8,
    }

    accelerator = Accelerator(
        mixed_precision = "fp16",
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
    B = 2
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
    T = 512
    model = GraphLLM(max_text_len=T,
                    max_max_new_tokens=32,
                    llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                    llm_frozen=False # set frozen to false so we can train with RL
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
    progress_bar = tqdm.tqdm(range(num_training_steps))
    best_val_loss = float('inf')
    # log_path = '../logs/graph_llm_no_text/'
    save_path = '../checkpoints/graph_llm_no_text/'

    ## TRAIN LOOP
    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            # grad accumulation
            with accelerator.accumulate(model):
                loss = model(batch)
                accelerator.backward(loss)
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                optimizer.step()
                optimizer.zero_grad()
            epoch_loss += loss.item()
            progress_bar.update(1)
            adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
        
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                loss = model(batch)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        accelerator.print(f"Epoch {epoch}/{args.num_epochs} | "
                    f"Train Loss: {epoch_loss / len(train_loader):.4f} | "
                    f"Validation Loss: {val_loss:.4f} | "
                    f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}", 
                    end="\r")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # only main process saves model
            if accelerator.is_main_process:
                _save_checkpoint(model, optimizer, epoch, args, save_path, is_best=True)

        # Early stopping if needed
        if epoch - best_epoch >= args.patience:
            accelerator.print(f"\nEarly stopping at epoch {epoch}")
            break
    
    accelerator.print("Training Complete")

if __name__ == '__main__':
    main()
