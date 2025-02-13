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

# ----------------
## SETUP FUNCTIONS
# ----------------
def fsdp_setup():
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl',
                                init_method='env://')
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)
    return local_rank, global_rank, world_size

# clean up fsdp
def fsdp_cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

def main():
    # -------
    ## CONFIG
    # -------
    local_rank, global_rank, world_size = fsdp_setup()

    args = parse_args_llama()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    seed_everything(42)

    if global_rank == 0:
        print(f"Running FSDP training on {world_size} GPUs.")

    # ------------
    ## DATALOADERS
    # ------------
    data_path = '../data/subgraphs/all'
    dataset = BiologicalDataset(data_path)
    idx_split = dataset.get_idx_split()

    # split datasets on idx
    train_dataset = Subset(dataset, idx_split['train'])
    val_dataset = Subset(dataset, idx_split['val'])

    # Use DistributedSampler so each GPU gets its share of the data.
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)

    # make dataloaders
    B = 2
    train_loader = DataLoader(train_dataset, 
                            batch_size=B,
                            sampler=train_sampler,
                            collate_fn=collate_fn)

    val_loader = DataLoader(val_dataset, 
                            batch_size=B,
                            sampler=val_sampler,
                            collate_fn=collate_fn)

    # -----------
    ## MODEL INIT
    # -----------
    T = 256
    model = GraphLLM(max_text_len=T,
                    max_max_new_tokens=32,
                    llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                    llm_frozen=True,
                    fsdp=True).to(local_rank) # args are defaulted in the class
    
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.float16,
        reduce_dtype=torch.float16,
        buffer_dtype=torch.float16,)

    # load graph enc + projector separately
    model.graph_encoder = FSDP(model.graph_encoder,
                               mixed_precision=mixed_precision_policy,
                               use_orig_params=False,
                               device_id=local_rank)

    model.projector = FSDP(model.projector,
                           mixed_precision=mixed_precision_policy,
                           use_orig_params=False,
                           device_id=local_rank)
    

    # --------------------
    ## OPTIMIZER & OPTIONS
    # --------------------
    # set optimizer
    params = [p for _, p in model.named_parameters() if p.requires_grad] # only update non-frozen params (graph encoder)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.95))

    # options
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm.tqdm(range(num_training_steps))
    best_val_loss = float('inf')
    # log_path = '../logs/graph_llm_no_text/'
    save_path = '../checkpoints/graph_llm_no_text/'

    ## TRAIN LOOP
    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            # Move all tensors in the batch to the correct device using non_blocking transfers.
            batch = {k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            loss = model(batch)
            loss.backward()
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            
            # grad steps is a hyprparameter
            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)
                optimizer.step()
            epoch_loss += loss.item()
            progress_bar.update(1)
        
        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in enumerate(val_loader):
                batch = {k: (v.to(local_rank, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                loss = model(batch)
                val_loss += loss.item()
            val_loss /= len(val_loader)
        
        if global_rank ==0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, save_path, is_best=True)
            best_epoch = epoch
        
        # Print training status (each process prints; you might want to limit this to rank 0)
        if global_rank == 0:
            print(f"Epoch {epoch}/{args.num_epochs} | Train Loss: {epoch_loss / len(train_loader):.4f} | "
                f"Validation Loss: {val_loss:.4f} | Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch}", end="\r")
            
        # Early stopping if needed
        if epoch - best_epoch >= args.patience:
            if dist.get_rank() == 0:
                print(f"\nEarly stopping at epoch {epoch}")
            break

        # clear cache
        torch.cuda.empty_cache()

    # --------
    ## CLEANUP
    # --------
    fsdp_cleanup()

if __name__ == '__main__':
    main()
