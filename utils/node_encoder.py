from numbers import Integral
import numpy as np

from .multiplex import Multiplex
from .rwr import rwr_encoding

def encode_nodes(nodes_to_encode: str | list[str],
                 mp: Multiplex,
                 out_dim: int = 1024,
                 base_encode_method: str = 'rwr',
                 up_encode_method: str = 'pad',
                 down_encode_method: None | str = None):

  def _get_allowed_values(x: list[str]) -> str:
    if len(x) == 1:
      return f'{x[0]}'
    else:
      return f"{', '.join([str(a) for a in x[:-1]])} or {x[-1]}"

  allowed_base_encode_methods = ['rwr']
  allowed_up_encode_method = ['pad']

  if not (isinstance(nodes_to_encode, str) or isinstance(nodes_to_encode, list)):
    raise TypeError("nodes_to_encode must be a 'str' or 'list[str]'")
  if not isinstance(mp, Multiplex):
    raise TypeError("mp must be a Multiplex")
  if not isinstance(out_dim, Integral):
    raise TypeError("out_dim must be ab integer")
  if out_dim <= 0:
    raise ValueError("out_dim must be positive")
  if not isinstance(base_encode_method, str):
    raise TypeError("base_encode_method must be a string")
  if base_encode_method not in allowed_base_encode_methods:
    raise ValueError(f'base_encode_method must be {_get_allowed_values(allowed_base_encode_methods)}')
  if up_encode_method not in allowed_up_encode_method:
    raise ValueError(f'up_encode_method must be {_get_allowed_values(allowed_up_encode_method)}')
  if down_encode_method is not None:
    raise ValueError('No methods are implemented for down_encoding')

  A = mp.adj_matrix()

  if base_encode_method == 'rwr':
    P = rwr_encoding(nodes_to_encode, A, mp.nodes, len(mp))

  # Check if encoding is smaller than desired output
  if P.shape[0] < out_dim:
    pad_width = out_dim - P.shape[0]
    P = np.pad(P, pad_width=((0,pad_width), (0,0)), mode='constant', constant_values=0)

  # Check if encoding is larger than desired output
  if P.shape[0] > out_dim:
    raise ValueError("Code is not set up for encodings larger than 'out_dim'")
  
  return P