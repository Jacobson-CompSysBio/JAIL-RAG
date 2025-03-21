import networkx as nx
import numpy as np
import pandas as pd
import os, sys, glob
import torch
from torch_geometric.data.data import Data

from textualize import textualize_edges, load_textualizer
from multiplex import *
from generate_split import generate_split
from node_encoder import encode_nodes
from utils import array_sampler

# relative path
sys.path.append('../')

def _get_node_idx_from_pair_idx(idx: int, num_nodes: int) -> tuple:
  u = int(idx / num_nodes)
  v = idx - u * num_nodes
  return u, v

def _get_max_pairs(num_nodes: int) -> int:
  return num_nodes * (num_nodes - 1)

def _get_max_combinations(num_nodes: int) -> int:
  return num_nodes * (num_nodes - 1) / 2

def generate_connection_data_mono(textualize,
                                  mp: Multiplex,
                                  output_dir: str,
                                  file_name: str,
                                  pt_obj_path: str, 
                                  use_node_id: bool = False,
                                  num_tests: int = -1,
                                  bal_pos_neg: bool = True) -> None:
  if len(mp) > 1:
    print("Skipping 'generate_connection_data_mono' - test is designed for monoplexes")
    return
  
  N = mp.layers[0]['graph'].number_of_nodes()
  # Calculate maximum number of pairs (i,j) where i != j
  max_pairs = N * (N - 1)

  # Limit the number of tests
  if num_tests == -1:
    print(f'Setting number of connection tests to {max_pairs}')
    num_tests = max_pairs
  if num_tests > max_pairs:
    print(f'Limiting number of connection tests to {max_pairs} to avoid duplication.')
    num_tests = max_pairs
  

  # Calculate number of possible positive and negative tests
  n_pos_tests = 2 * mp.layers[0]['graph'].number_of_edges()
  n_neg_tests = max_pairs - n_pos_tests
  
  # Check if balancing positive and negative tests
  if bal_pos_neg:
    print('Balancing positive and negative tests')
    # Check is tests can be balanced while meeting 'num_tests' requirement
    min_pos_neg = min([n_pos_tests, n_neg_tests])

    if 2 * min_pos_neg > num_tests:
      print(f'Cannot balance positive and negative tests while generating {num_tests} tests.')
      
      min_pos_neg = int(num_tests/2)
      print(f'Reducing number of tests to {2*min_pos_neg}')

    n_pos_tests = min_pos_neg
    n_neg_tests = min_pos_neg
  else:
    # Reduce the number of positive and negative tests while keeping same ratio
    n_pos_tests = int(n_pos_tests / max_pairs * num_tests)
    n_neg_tests = num_tests - n_pos_tests

  print(f'Creating {n_pos_tests} positive tests and {n_neg_tests} negative tests')

  data = pd.DataFrame(None, columns=['question','scope','label','desc','graph'])

  # Add positive tests
  edges = list(mp[0]['graph'].edges())
  edges_idx = np.array([i for i in range(2*len(edges))])
  pos_sampler = array_sampler(edges_idx)
  n_tests = 0
  while n_tests < n_pos_tests:
    idx = pos_sampler.sample()
    if mp[0]['graph'].is_directed():
      u, v = edges[idx]
    else:
      if idx >= len(edges):
        idx = idx % len(edges)
        v, u = edges[idx]
      else:
        u, v = edges[idx]

    if use_node_id:
      u = mp.nodes.index(u)
      v = mp.nodes.index(v)

    question = f'Is there an edge between nodes {u} and {v}?'
    scope = 'all'
    label = 'yes'
    desc =  ' '
    graph = pt_obj_path
    data.loc[len(data)] = [question, scope, label, desc, graph]

    n_tests += 1

  # Add negative tests
  pair_idxs = np.array([i for i in range(N*N)])
  sampler = array_sampler(pair_idxs)

  nodes = mp.nodes
  n_tests = 0
  while n_tests < n_neg_tests:
    pair_idx = sampler.sample()

    u_idx = int(pair_idx / N)
    v_idx = pair_idx % N
    u = nodes[u_idx]
    v = nodes[v_idx]

    if u == v:
      pass
    elif mp[0]['graph'].has_edge(u, v):
      pass
    else:
      if use_node_id:
        u = nodes.index(u)
        v = nodes.index(v)
    
      question = f'Is there an edge between nodes {u} and {v}?'
      scope = 'all'
      label = 'no'
      desc =  ' '# textualize(mp)
      graph = pt_obj_path
      data.loc[len(data)] = [question, scope, label, desc, graph]
      n_tests += 1

  data = data.sample(frac=1)

  os.makedirs(output_dir, exist_ok=True)
  output_file = os.path.join(output_dir, file_name)
  add_headers = not os.path.exists(output_file)
  data.to_csv(output_file, sep='\t', index=False, mode='a', header=add_headers)

  return n_pos_tests + n_neg_tests

if __name__ == '__main__':

  # change this for different textualization types
  textualizer_name = 'edges'

  # CHANGE PATHS
  base_dir = '../data/subgraphs'

  # Get name of all files in 'base_dir
  graph_files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]

  # Init number of tests
  num_tests = 0
  textualize = load_textualizer[textualizer_name]

  # Loop through each graph
  for i, graph_file in enumerate(graph_files):
    graph_file = os.path.join(base_dir, graph_file)
    flist = os.path.join(base_dir, 'mono_flist.tsv')
    with open(flist, 'w') as fp:
      fp.write(f'{graph_file}\tunknown\n')
    
    mp = Multiplex(flist)

    N = mp.layers[0]['graph'].number_of_nodes()
    dir = 'all'
    output_dir = os.path.join(base_dir, f'{dir}')
    graph_dir = os.path.join(output_dir, 'graphs')
    os.makedirs(graph_dir, exist_ok=True)
      
    node_encoding = encode_nodes(mp.nodes, mp)
    edge_index = torch.LongTensor([mp.src(), mp.dst()])
    data = Data(x=node_encoding, edge_index=edge_index, num_nodes=N)
    pt_file = os.path.join(graph_dir, f'{i}.pt')
    torch.save(data, pt_file)

    desired_test = int(mp.layers[0]['graph'].number_of_edges() / 3)
    print(f'Desired number of tests: {desired_test}')
    num_tests += generate_connection_data_mono(textualize, mp, output_dir, 'train_dev.tsv', pt_file, use_node_id=True, num_tests=desired_test)
  split_path = os.path.join(output_dir, 'split')
  generate_split(num_tests, split_path)