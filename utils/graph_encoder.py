# all of our graph embedding methods
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv

class GCN(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()

        # inital conv layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))

        # initial bn layer
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # hidden conv + bn stack
        for _ in range(num_layers = 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # final conv 
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    # forward pass - since these are the only trainable modules in the paper
    def forward(self, x, adj_t, edge_attr):

        # loop through all convs except last
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # final layer
        x = self.convs[-1](x, adj_t)

        # return input graph + edge attributes
        return x, edge_attr

class GraphTransformer(nn.Module):
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GraphTransformer, self).__init__()

        # initial transformer + bn
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, 
                                          out_channels=hidden_channels//num_heads, 
                                          heads=num_heads, 
                                          edge_dim=in_channels, 
                                          dropout=dropout))
        
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))

        # hidden transformer + bn
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels,
                                              out_channels=hidden_channels//num_heads,
                                              heads=num_heads,
                                              edge_dim=in_channels,
                                              dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        
        # final transformer + drop
        self.convs.append(TransformerConv(in_channels=hidden_channels,
                                          out_channels=out_channels//num_heads,
                                          edge_dim=in_channels,
                                          dropout=dropout))
        self.dropout = dropout
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i]
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr
        

class GAT():
    
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr

# define load gnn model dict that call respective class
load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gt': GraphTransformer
}
    