from sklearn.preprocessing import normalize
import numpy as np
from scipy import sparse
import math

def column_norm(X):
  return sparse.csc_array(normalize(X, norm='l1', axis=0))

def get_init_prob(seed: str, nodes: list[str], L: int) -> np.ndarray:
  idx = nodes.index(seed)
  p = np.zeros(len(nodes))
  p[idx] = 1.0 / L
  return np.tile(p, L)

def random_walk_restart(M: sparse.csr_array,
                        p0: np.ndarray,
                        r: float,
                        L: int,
                        tau: None | list[float] = None,
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

def rwr_encoding(seeds: str | list[str],
                 adj,
                 node_list: list[str],
                 L: int,
                 restart_prob: float = 0.7,
                 tau: None | list[float] = None,
                 threshold: float = 1e-10):
  if isinstance(seeds, str):
    seeds = [seeds]
  
  # Check that all seeds are in the node_list
  in_node_list = [ seed not in node_list for seed in seeds]
  if sum(in_node_list) > 0:
    raise ValueError(f'Not all seeds are in the node_list')

  # Create transition matrix for Markov process by normlaizing each column
  # in the adjacency matrix
  M = column_norm(adj)

  # Initialize encoding
  P = np.empty((M.shape[0], len(seeds)))

  for j, seed in enumerate(seeds):
    # Initialize the similarity matrix
    p0 = get_init_prob(seed, node_list, L)

    # Perform random walk with restart until probability vector is stable
    p_stable = random_walk_restart(M, p0, restart_prob, L, tau, threshold)

    # Normalize vector
    p_stable = p_stable / np.sum(p_stable)

    P[:,j] = p_stable
  
  return P



# def generate_similarity_matrix(seed: str,
#                                adj,
#                                nodes: list[str],
#                                restart_prob: float,
#                                L: int,
#                                tau: None | list[float] = None,
#                                threshold: float = 1e-10):
#   # Create transition matrix fro Markov process by normlaizing each column
#   # in the adjacency matrix
#   M = column_norm(adj)

#   # Initialize the similarity matrix
#   p0 = get_init_prob(seed, nodes, L)

#   # Perform random walk with restart until probability vector is stable
#   p_stable = random_walk_restart(M, p0, restart_prob, L, tau, threshold)

#   # Normalize vector
#   p_stable = p_stable / np.sum(p_stable)
  
#   return p_stable