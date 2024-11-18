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
    L = len(self)

    # Check if multiplex is empty
    if L == 0:
      return np.ndarray(0)
    
    # Check if multiplex contains a single layer
    if L == 1:
      return nx.adjacency_matrix(self.layers[0]['graph'], self._nodes)
    
    # Check that delta is valid
    if delta < 0 or delta > 1:
      raise ValueError('delta should be in [0,1]')

    N = self.num_nodes
    eye = delta / (L-1) * sparse.identity(N, format='csr')
    
    def get_nodes_in_layer(layer_idx: int) -> list[str]:
      return [n for n in self._nodes if n in list(self.layers[layer_idx]['graph'].nodes())]
  
    blocks = [[(1-delta) * nx.to_scipy_sparse_array(self.layers[i]['graph'], get_nodes_in_layer(0)) if l == i else eye for l in range(len(self))] for i in range(len(self))]

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

  @property
  def src(self, layer_idx: int = -1) -> list:
    if layer_idx < -1 or layer_idx >= len(self):
      raise ValueError(f'layer_idx must be between -1 and {len(self)-1}')
    
    edges = []
    if layer_idx == -1:
      for l in self.layers:
        edges += list(self.layers[l]['graph'].edges())
    else:
      edges += list(self.layers[layer_idx]['graph'].edges())
    
    return [e[0] for e in edges]
  
  @property
  def dst(self, layer_idx: int = -1) -> list:
    if layer_idx < -1 or layer_idx >= len(self):
      raise ValueError(f'layer_idx must be between -1 and {len(self)-1}')
    
    edges = []
    if layer_idx == -1:
      for l in self.layers:
        edges += list(self.layers[l]['graph'].edges())
    else:
      edges += list(self.layers[layer_idx]['graph'].edges())
    
    return [e[1] for e in edges]
  