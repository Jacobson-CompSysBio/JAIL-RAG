from torch_geometric.data import Batch

def collate_fn(original_batch):
    # make dict to hold out batch
    batch = {}

    # iterate over keys in input batch
    for k in original_batch[0].keys:
        # set out batch key to hold a list of the values of the input batch
        batch[k] = [d[k] for d in original_batch]
    
    # if the out batch contains a 'graph' key, use the PyG Batch object instead of a list
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    
    return batch