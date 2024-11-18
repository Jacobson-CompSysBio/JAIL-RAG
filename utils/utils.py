import numpy as np
from numpy.random import Generator, PCG64

def get_allowed_values_as_str(x: list) -> str:
  print(type(x))
  if len(x) == 1:
    return f'{x[0]}'
  else:
    return f"{', '.join([str(a) for a in x[:-1]])} or {x[-1]}"

class array_sampler:
  def __init__(self, arr: np.ndarray) -> None:
    self.arr = np.array(arr)
    self.num_values = self.arr.size
    self.rng = Generator(PCG64())
  
  def sample(self):
    if self.num_values == 0:
      raise ValueError('All values from array have been selected')
    
    if self.num_values == 1:
      self.num_values -= 1
      return self.arr[0]

    idx = self.rng.choice(self.num_values-1)
    val = self.arr[idx]
    self.arr[idx] = self.arr[self.num_values-1]
    self.num_values -= 1

    return val