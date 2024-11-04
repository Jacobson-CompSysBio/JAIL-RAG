import networkx as nx
import numpy as np
import os
from .textualize import *
from .multiplex import *
from .generate_split import generate_split

def _get_node_idx_from_pair_idx(idx: int, num_nodes: int) -> tuple[int, int]:
  u = int(idx / num_nodes)
  v = idx - u * num_nodes
  return u, v

def _get_max_pairs(num_nodes: int) -> int:
  return num_nodes * (num_nodes - 1)

def generate_connection_data_mono(textualize, mp: Multiplex, num_tests: int, output_file: str) -> None:
  if len(mp) > 1:
    print("Skipping 'generate_connection_data_mono' - test is designed for monoplexes")
    return
    
  nodes = mp.nodes
  max_pairs = _get_max_pairs(mp.num_nodes)

  if num_tests >= max_pairs:
    print(f'Limiting number of connection tests to {max_pairs} to avoid duplication.')
    num_tests = max_pairs

  pairs_idx = np.random.choice(range(max_pairs), num_tests, replace=False)

  if not os.path.isfile(output_file):
    with open(output_file, "w") as fp:
      fp.write('question\tlabel\tdesc\n')
      
  with open(output_file, "a") as fp:
    for i in pairs_idx:
      u_idx, v_idx = _get_node_idx_from_pair_idx(i, mp.num_nodes)
      u = nodes[u_idx]
      v = nodes[v_idx]

      if mp[0]['graph'].has_edge(u,v):
        label = 'yes'
      else:
        label = 'no'
      
      question = f'Is there an edge between nodes {u} and {v}?'
      desc = textualize(mp)

      fp.write(f'{question}\t{label}\t{desc}\n')

def generate_path_data_mono(textualize, mp: Multiplex, num_tests: int, output_file: str) -> None:
  if len(mp) > 1:
    print("Skipping 'generate_path_data_mono' - test is designed for monoplexes")
    return
  
  nodes = mp.nodes
  max_pairs = _get_max_pairs(mp.num_nodes)

  if num_tests >= max_pairs:
    print(f'Limiting number of path tests to {max_pairs} to avoid duplication.')
    num_tests = max_pairs

  pairs_idx = np.random.choice(range(max_pairs), num_tests, replace=False)

  if not os.path.isfile(output_file):
    with open(output_file, "w") as fp:
      fp.write('question\tlabel\tdesc\n')
  
  with open(output_file, "a") as fp:
    for i in pairs_idx:
      u_idx, v_idx = _get_node_idx_from_pair_idx(i, mp.num_nodes)
      u = nodes[u_idx]
      v = nodes[v_idx]

      if nx.has_path(mp[0]['graph'], u, v):
        label = 'yes'
      else:
        label = 'no'
      
      question = f'Is there a path between nodes {u} and {v}?'
      desc = textualize(mp)

      fp.write(f'{question}\t{label}\t{desc}\n')

def generate_shortest_path_data_mono(textualize, mp: Multiplex, num_tests: int, output_file: str) -> None:
  if len(mp) > 1:
    print("Skipping 'generate_shortest_path_data_mono' - test is designed for monoplexes")
    return
  
  nodes = mp.nodes
  max_pairs = _get_max_pairs(mp.num_nodes)

  if num_tests >= max_pairs:
    print(f'Limiting number of shortest path tests to {max_pairs} to avoid duplication.')
    num_tests = max_pairs

  pairs_idx = np.random.choice(range(max_pairs), num_tests, replace=False)

  if not os.path.isfile(output_file):
    with open(output_file, "w") as fp:
      fp.write('question\tlabel\tdesc\n')
  
  with open(output_file, "a") as fp:
    for i in pairs_idx:
      u_idx, v_idx = _get_node_idx_from_pair_idx(i, mp.num_nodes)
      u = nodes[u_idx]
      v = nodes[v_idx]

      if nx.has_path(mp[0]['graph'], u, v):
        label = nx.shortest_path(mp[0]['graph'], u, v)
      else:
        label = 'There is no path'
      
      question = f'What is the shortest path between nodes {u} and {v}?'
      desc = textualize(mp)

      fp.write(f'{question}\t{label}\t{desc}\n')

def generate_data_mono(textualizer_name: str, output_path: str, flist: str, num_tests: int) -> None:
  textualize = load_textualizer[textualizer_name]
  mp = Multiplex(flist)

  output_file = os.path.join(output_path, 'train_dev.tsv')

  num_qa = 0

  generate_connection_data_mono(textualize, mp, num_tests, output_file)
  num_qa += num_tests

  generate_path_data_mono(textualize, mp, num_tests, output_file)
  num_qa += num_tests

  generate_shortest_path_data_mono(textualize, mp, num_tests, output_file)
  num_qa += num_tests

  split_path = os.path.join(output_path, 'split')
  generate_split(num_qa, split_path)
