import networkx as nx
import pandas as pd
from scipy import sparse
import numpy as np

class Multiplex:

  """Class for multiplex"""
  
  def __init__(self, flist: str = None):
    self.layers = []
    self._nodes = []

    if flist is not None:
      layer_info = pd.read_csv(flist, sep='\t', header=None)

      if layer_info.shape[1] != 2:
        raise ValueError('flist file must contain two tab-seperated columns.')
        
      for i in layer_info.index:
        g = nx.read_edgelist(layer_info.iloc[i,0],
                             create_using=nx.Graph,
                             nodetype=str,
                             data=(('weight', float),))
        self.add_layer(g, layer_info.iloc[i,1])

  def __len__(self):
    """Return the number of layers in the multiplex"""
    return len(self.layers)
    
  def __getitem__(self, index):
    return self.layers[index]

  def add_layer(self, g: nx.Graph, layer_name: str):
    self.layers.append({
      'graph': g,
      'layer_name': layer_name,
    })

    self._nodes = list(set(self._nodes).union(g.nodes))
    self._nodes.sort()
  
  def adj_matrix(self, delta: float = 0.5):
    # Check that delta is valid
    if delta < 0 or delta > 1:
      raise ValueError('delta should be in [0,1]')
    
    L = len(self)

    # Check if multiplex is empty
    if L == 0:
      return np.ndarray(0)
    
    # Check if multiplex contains a single layer
    if L == 1:
      return nx.adjacency_matrix(self.layers[0]['graph'], self._nodes)
    
    N = self.num_nodes
    eye = delta / (L-1) * sparse.identity(N, format='csr')
    
    def get_nodes_in_layer(layer_idx: int) -> list[str]:
      return [n for n in self._nodes if n in list(self.layers[layer_idx]['graph'].nodes())]
  
    blocks = [[(1-delta) * nx.to_scipy_sparse_array(self.layers[i]['graph'], get_nodes_in_layer(0)) if l == i else eye for l in range(L)] for i in range(L)]

    return sparse.block_array(blocks, format='csr')
  
  @property
  def num_nodes(self) -> int:
    """Get number of nodes"""
    if len(self.layers) == 0:
      return 0
    else:
      return len(self.nodes)
  
  @property
  def nodes(self) -> list:
    """Get list of nodes"""
    return self._nodes

  def sort_edges(self, edges, nodelist) -> list:
    sorted_edges = []

    for e in edges:
      if nodelist.index(e[0]) < nodelist.index(e[1]):
        sorted_edges.append((e[0],e[1]))
      else:
        sorted_edges.append((e[1],e[0]))
    
    sorted_edges.sort()

    return sorted_edges

  def src(self, layer_idx: int = -1) -> list:
    if not isinstance(layer_idx, int):
      raise TypeError('layer_idx must be an integer')
    if layer_idx < -1 or layer_idx >= len(self):
      if len(self) == 1:
        raise ValueError('layer_idx must be -1 or 0')
      else:
        raise ValueError(f'layer_idx must be -1 or an integer between 0 and {len(self)-1}')
    
    if len(self) == 0:
      return []
    
    edges = []
    if layer_idx == -1:
      for layer in self.layers:
        edges += list(layer['graph'].edges())
    else:
      edges += list(self.layers[layer_idx]['graph'].edges())
    
    edges = self.sort_edges(edges, self._nodes)
    src = [e[0] for e in edges]

    return [self._nodes.index(s) for s in src]
  
  def dst(self, layer_idx: int = -1) -> list:
    if layer_idx < -1 or layer_idx >= len(self):
      raise ValueError(f'layer_idx must be between -1 and {len(self)-1}')
    
    if len(self) == 0:
      return []
    
    edges = []
    if layer_idx == -1:
      for layer in self.layers:
        edges += list(layer['graph'].edges())
    else:
      edges += list(self.layers[layer_idx]['graph'].edges())
    
    edges = self.sort_edges(edges, self._nodes)
    dst = [e[1] for e in edges]

    return [self.nodes.index(d) for d in dst]
  