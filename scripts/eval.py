import os, sys
import wandb
from tqdm.notebook import tqdm
import transformers
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, Subset
from importlib import reload
from torch_scatter import scatter
from transformers import pipeline
from IPython.display import clear_output

sys.path.append('../')

from utils import preprocess as pp
from utils.evaluate import eval_funcs, normalize
from utils.collate import collate_fn 
from utils.ckpt import _reload_best_model
from utils.graph_llm import GraphLLM
from utils.llm import LLM
from utils.multiplex import Multiplex
from utils.textualize import *
from utils.bio_graphs import BiologicalDataset
from utils.seed import seed_everything

# -------
# OPTIONS
# -------
verbose = True
seed = 42
seed_everything(seed)

batch_size = 1
data_path = '../data/subgraphs/all'
model_path = '../checkpoints/graph_llm_fsdp/epoch_2_best.pth' # REPLACE WITH BEST MODEL PATH
eval_path = '../logs/eval/graph_llm_fsdp/'

# --------------------
# DATASET / DATALOADER
# --------------------
dataset = BiologicalDataset(data_path)
idx_split = dataset.get_idx_split()

# split datasets on idx
test_dataset = Subset(dataset, idx_split['test'])

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          drop_last=True,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn)

# ----------
# LOAD MODEL
# ----------
base = GraphLLM(max_text_len=512,
                     max_max_new_tokens=32,
                     max_memory=[80, 80],
                     llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                     llm_frozen=True,
                     revision="main") # args are defaulted in the class

model = _reload_best_model(base, model_path)

# --------
# EVALUATE
# --------
# options
loader = test_loader

# set to eval
model.model.generation_config.pad_token_id = model.tokenizer.pad_token_id
model.eval()

n_correct = 0
i = 0
# loop through dataloader
with torch.no_grad():
    for batch in tqdm(loader):
        out = model.inference(batch)
        pred = out['pred']
        actual = out['label']
        # test accuracy
        for p, a in zip(pred, actual):
            p_ans, p_think = normalize(p)
            a = str(a)
            if verbose:
                print(p_think)
                print(p_ans)
                print(a)
                print()
            if a in p_ans:
                n_correct += 1
                if verbose:
                    print("Correct!")
                    print()
            else:
                if verbose:
                    print("Incorrect :(")
                    print()
            i += 1
        print(f"Accuracy: {n_correct/i:.2%} | {n_correct}/{i}", end='\r')
    acc = n_correct / i
    print(f"Accuracy: {acc:.2%} | {n_correct}/{i}")

# ------------------
# PRINT EXAMPLE EVAL
# ------------------
with open(f'{eval_path}/eval.csv', 'w') as f:
    f.write("actual,prediction,reasoning")
    rand_idx = torch.randint(0, len(test_dataset), (1,)).item()
    example_batch = test_dataset[rand_idx]
    with torch.no_grad():
        out = model.inference(example_batch)
    pred = out['pred']
    actual = out['label']
    for p, a in zip(pred, actual):
        p_ans, p_think = normalize(p)
        a = str(a)
        f.write(f"{a},{p_ans},{p_think}")