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
# from utils.model import load_model, llama_model_path
from utils.evaluate import eval_funcs
from utils.config import parse_args_llama
from utils.ckpt import _save_checkpoint, _reload_best_model
from utils.collate import collate_fn
from utils.seed import seed_everything
from utils.lr_schedule import adjust_learning_rate