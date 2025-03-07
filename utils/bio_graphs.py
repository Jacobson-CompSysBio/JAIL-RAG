import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------
## NORMAL DATASET
# ---------------
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
      'desc': 'A question with a yes/no answer is provided along with a graph. Answer the question based on the graph. Provide reasoning inside of <think></think> tags and the answer inside of <answer></answer> tags.',
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