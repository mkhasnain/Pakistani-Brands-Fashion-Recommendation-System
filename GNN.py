import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import torch_geometric.transforms as T

# Prepare the interaction data as edge index
edges = torch.tensor(interactions_df[['user_id', 'item_id']].values).t().contiguous()
edge_index = torch.cat([edges, edges[[1, 0]]], dim=1)  # Making it undirected

# Create the node features and labels (initially just using identity matrix as features)
num_nodes = num_users + num_items
node_features = torch.eye(num_nodes)

# Prepare the edge weights
edge_weight = torch.ones(edge_index.size(1))

data = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight)
