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

class GCNRecommendationModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNRecommendationModel, self).__init__()
        self.conv1 = GCNConv(in_channels, 64)
        self.conv2 = GCNConv(64, out_channels)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x
