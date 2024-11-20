from sklearn.preprocessing import normalize
import numpy as np
from scipy import sparse
import math
from scipy.stats import gmean
from utils import get_allowed_values_as_str

def column_norm(X):
  return sparse.csc_array(normalize(X, norm='l1', axis=0))

def get_init_prob(seed: str, nodes: list, L: int) -> np.ndarray:
  idx = nodes.index(seed)
  p = np.zeros(len(nodes))
  p[idx] = 1.0 / L
  return np.tile(p, L)

def random_walk_restart(M: sparse.csr_array,
                        p0: np.ndarray,
                        r: float,
                        L: int,
                        tau: None = None,
                        threshold: float = 1e-10) -> np.ndarray:
  # Check inputs
  if r < 0 or r > 1:
    raise ValueError('restart_prob must be between 0 and 1')
  
  if threshold <= 0:
    raise ValueError('threshold must be positive')
  
  if type(tau) is list:
    if len(tau) != L:
      raise ValueError('tau must contain L elements')
    if not math.isclose(np.sum(tau)/L, 1.0):
      raise ValueError('elements of tau must sum to L')
    tau = np.array(tau)
  elif tau is None:
    tau = np.empty(L)
    tau.fill(1.0/L)
  else:
    raise TypeError('tau must be None or list[float]')
  
  # Adjust initial probability vector based on tau (probability of restarting on a specific layer)
  # Multiple the first N elements by tau[0], the next N elements by tau[1], etc.
  N = int(M.shape[0]/L)
  tau = np.repeat(tau, N)
  p0 = tau * p0
  p0 = p0 / np.sum(p0)

  p_rs = r * p0
  p = p0
  residue = 1.0

  while (residue >= threshold):
    p_old = p
    p = (1-r) * M.dot(p) + p_rs
    residue = np.max(np.linalg.norm(p - p_old))

  return p

def _geometric_mean(X: np.ndarray, L: int, N: int):
  # Check inputs
  if X.shape[0] != N*L:
    raise ValueError('The number of rows in X must equak N*L')
    
  mean = np.empty(N)

  for i in range(N):
    idx = np.array([i + l*N for l in range(L)])
    mean[i] = gmean(X[idx])
    
  return mean

def _arithemtic_mean(X: np.ndarray, L: int, N: int):
  # Check inputs
  if X.shape[0] != N*L:
    raise ValueError('The number of rows in X must equak N*L')
    
  mean = np.empty(N)

  for i in range(N):
    idx = np.array([i + l*N for l in range(L)])
    mean[i] = np.mean(X[idx])
    
  return mean
  
def _sum_scores(X: np.ndarray, L: int, N: int):
  # Check inputs
  if X.shape[0] != N*L:
    raise ValueError('The number of rows in X must equak N*L')
  
  mean = np.empty(N)

  for i in range(N):
    idx = np.array([i + l*N for l in range(L)])
    mean[i] = sum(X[idx])
    
  return mean

def rwr_encoding(seeds: str,
                 adj,
                 node_list: list,
                 L: int,
                 restart_prob: float = 0.7,
                 mean_type: str = None,
                 tau: None = None,
                 threshold: float = 1e-10):
  if isinstance(seeds, str):
    seeds = [seeds]
  
  # Check that all seeds are in the node_list
  in_node_list = [ seed not in node_list for seed in seeds]
  if sum(in_node_list) > 0:
    raise ValueError(f'Not all seeds are in the node_list')
  
  if adj.shape[0] != adj.shape[1]:
    raise ValueError('Adjacency matrix must be square')
  
  N = len(node_list)
  if N*L != adj.shape[0]:
    raise ValueError('adj must have a dimension of N*L along each axis')
  
  allowed_mean_types = ['Geometric', 'Arithmetic', 'Sum']
  if mean_type is not None and mean_type not in allowed_mean_types:
    raise ValueError(f'mean_type must be None, {get_allowed_values_as_str(allowed_mean_types)}')

  # Create transition matrix for Markov process by normlaizing each column
  # in the adjacency matrix
  M = column_norm(adj)

  # Initialize encoding
  if mean_type is None:
    P = np.empty((N, len(seeds)))
  else:
    P = np.empty((M.shape[0], len(seeds)))

  for j, seed in enumerate(seeds):
    # Initialize the similarity matrix
    p0 = get_init_prob(seed, node_list, L)

    # Perform random walk with restart until probability vector is stable
    p_stable = random_walk_restart(M, p0, restart_prob, L, tau, threshold)

    # Summarize the embedding values for each layer
    if L > 1 and mean_type is not None:
      if mean_type == 'Geometric':
        p_stable = _geometric_mean(p_stable, L, N)
      elif mean_type == "Arithmetic":
        p_stable = _arithemtic_mean(p_stable, L, N)
      else:
        p_stable = _sum_scores(p_stable, L, N)
    
    # Normalize vector
    p_stable = p_stable / np.sum(p_stable)

    P[:,j] = p_stable
  
  return P
