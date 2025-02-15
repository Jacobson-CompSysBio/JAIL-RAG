import pandas as pd
import torch
from torch.utils.data import Dataset

# PATH = '../data/DREAM4_gold_standards/shortest_path_node_id'

class BiologicalDataset(Dataset):
  def __init__(self, path: str):
    super().__init__()

    self.path = path
    self.text = pd.read_csv(f'{self.path}/train_dev.tsv', sep='\t')
    self.prompt = None
    self.graph = None
    self.graph_type = 'Biological Graph'

  def __len__(self):
    """Return the len of the dataset."""
    return len(self.text)
  
  def __getitem__(self, index: int):
    text = self.text.iloc[index]
    graph = torch.load(text['graph'], weights_only=False)
    graph.x = torch.tensor(graph.x.T, dtype=torch.float32)
    graph.edge_index = graph.edge_index.type(torch.int64)

    return {
      'id': index,
      'question': text['question'],
      'scope': text['scope'],
      'label': text['label'],
      'desc': 'You will be given a biological graph and a question. Provide an answer of YES of NO based on the question and the given input graph. Explain your reasoning, and surround it with with <think> and </think> tags. Then provide an answer, surrounded in <answer> and </answer> tags. ',
      'graph': graph,
    }
  
  def get_idx_split(self):
    # Load the saved indices
    with open(f'{self.path}/split/train_indices.txt', 'r') as file:
      train_indices = [int(line.strip()) for line in file]

    with open(f'{self.path}/split/val_indices.txt', 'r') as file:
      val_indices = [int(line.strip()) for line in file]

    with open(f'{self.path}/split/test_indices.txt', 'r') as file:
      test_indices = [int(line.strip()) for line in file]

    return {'train': train_indices, 'val': val_indices, 'test': test_indices}

