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

model = GCNRecommendationModel(in_channels=num_nodes, out_channels=32)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    user_item_preds = out[interactions_df['user_id']] @ out[interactions_df['item_id']].t()
    loss = criterion(user_item_preds.flatten(), torch.FloatTensor(interactions_df['rating']))
    loss.backward()
    optimizer.step()
    return loss.item()

# Training loop
num_epochs = 20
for epoch in range(num_epochs):
    loss = train()
    print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

def recommend_for_user_gcn(user_id, num_recommendations=5):
    model.eval()
    with torch.no_grad():
        out = model(data)
    
    user_embedding = out[user_id].unsqueeze(0)
    item_embeddings = out[num_users:num_users + num_items]
    
    scores = user_embedding @ item_embeddings.t()
    scores = scores.squeeze().numpy()
    top_item_indices = scores.argsort()[-num_recommendations:][::-1]
    recommended_items = df.iloc[top_item_indices].copy()
    
    # Inverse transform the categorical columns
    for column in label_encoders:
        recommended_items[column] = label_encoders[column].inverse_transform(recommended_items[column])
    
    return recommended_items

# Example usage: Get 5 recommendations for user 0
recommendations = recommend_for_user_gcn(0, 5)
print(recommendations)
