import os, sys
import json
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
from utils.evaluate import eval_funcs
from utils.collate import collate_fn 
from utils.ckpt import _reload_best_model
from utils.graph_llm import GraphLLM
from utils.llm import LLM
from utils.multiplex import Multiplex
from utils.textualize import *
from utils.bio_graphs import BiologicalDataset
from utils.seed import seed_everything

# ----------------------------------------------------------
# OPTIONS
seed = 42
seed_everything(seed)

batch_size = 1
data_path = '../data/DREAM4_gold_standards/connections_node_label'
model_path = '../checkpoints/test_run/epoch_5_best.pth' # REPLACE WITH BEST MODEL PATH
eval_path = '../logs/train/eval.json'

# ----------------------------------------------------------
# DATASET / DATALOADER

dataset = BiologicalDataset(data_path)
idx_split = dataset.get_idx_split()

# split datasets on idx
test_dataset = [dataset[i] for i in idx_split["test"]]

test_loader = DataLoader(test_dataset, 
                          batch_size=batch_size,
                          drop_last=True,
                          pin_memory=True,
                          shuffle=True,
                          collate_fn=collate_fn)

# ----------------------------------------------------------
# LOAD MODEL
base = GraphLLM(max_text_len=512,
                     max_max_new_tokens=32,
                     max_memory=[80, 80],
                     llm_model_path='meta-llama/Meta-Llama-3-8B-Instruct',
                     llm_frozen='True',
                     revision="main") # args are defaulted in the class

model = _reload_best_model(base, model_path)

# ----------------------------------------------------------
# EVALUATE
model.eval()
progress_bar_test = tqdm(range(len(test_loader)))
with open(eval_path, "w") as f:
    for _, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            df = pd.DataFrame(output)
            for _, row in df.iterrows():
                f.write(json.dumps(dict(row)) + "\n")
        progress_bar_test.update(1)

# Step 5. Post-processing & Evaluating
acc = eval_funcs[dataset](eval_path)
print(f'Test Acc {acc}')
# wandb.log({'Test Acc': acc})
