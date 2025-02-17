import os, sys
import wandb
from tqdm.auto import tqdm
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
verbose = False
seed = 42
seed_everything(seed)

batch_size = 64
data_path = '../data/subgraphs/all'
model_path = '../checkpoints/graph_llm_fsdp' # REPLACE WITH BEST MODEL PATH
eval_path = '../logs/eval/graph_llm_fsdp'

# --------------------
# DATASET / DATALOADER
# --------------------
dataset = BiologicalDataset(data_path)
idx_split = dataset.get_idx_split()

# split datasets on idx
test_dataset = Subset(dataset, idx_split['test'])

loader = DataLoader(test_dataset, 
                    batch_size=batch_size,
                    drop_last=True,
                    pin_memory=True,
                    shuffle=True,
                    collate_fn=collate_fn)

# ----------
# LOAD MODEL
# ----------
base = GraphLLM(max_text_len=256,
                max_max_new_tokens=512,
                max_memory=[80, 80],
                llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                llm_frozen=True,
                fsdp=False,
                revision="main") # args are defaulted in the class

# unwrap model
model = accelerator.unwrap_model(base)
model = model.load_state_dict(torch.load(model_path))

# --------
# EVALUATE
# --------
# set to eval
model.eval()

n_correct = 0
i = 0
pbar = tqdm(total=len(test_dataset))

# loop through dataloader
with torch.no_grad():
    for batch in loader:
        out = model.inference(batch)
        pred = out['pred']
        actual = out['label']

        # Process the batch predictions. Using a simple loop here since normalization involves string operations.
        for p, a in zip(pred, actual):
            p_ans, p_think = normalize(p)
            if verbose:
                print(p_think)
                print(p_ans)
                print(a)
                print()
                if str(a) in p_ans:
                    print("Correct!\n")
                else:
                    print("Incorrect :(\n")
            if str(a) in p_ans:
                n_correct += 1
        i += len(pred)
        pbar.update(len(pred))
pbar.close()

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