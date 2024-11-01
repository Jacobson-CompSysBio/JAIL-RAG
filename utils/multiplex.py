import networkx as nx
import pandas as pd

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
