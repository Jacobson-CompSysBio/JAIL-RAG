import pandas as pd
import torch
from torch.utils.data import Dataset

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
    graph = torch.load(f'{self.path}/graphs/{index}.pt')

    return {
      'id': index,
      'question': text['question'],
      'label': text['label'],
      'desc': text['desc'],
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
