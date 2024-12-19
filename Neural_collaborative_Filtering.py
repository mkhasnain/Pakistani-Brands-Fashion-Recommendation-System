import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# Load the dataset
df = pd.read_csv('clothes_dataset.csv')

# Encode categorical variables
label_encoders = {}
for column in df.columns:
    if df[column].dtype == 'object':
        label_encoders[column] = LabelEncoder()
        df[column] = label_encoders[column].fit_transform(df[column])

# Generate user-item interaction data (this is synthetic for the sake of the example)
# Assume that we have 100 users and each user has rated 20 random items
num_users = 100
num_items = df.shape[0]
interactions = []

for user_id in range(num_users):
    rated_items = np.random.choice(num_items, 20, replace=False)
    for item_id in rated_items:
        interactions.append((user_id, item_id, np.random.randint(1, 6)))  # Ratings between 1 and 5

# Create a DataFrame for interactions
interactions_df = pd.DataFrame(interactions, columns=['user_id', 'item_id', 'rating'])

# Define the custom dataset
class InteractionDataset(Dataset):
    def __init__(self, interactions_df):
        self.user_ids = interactions_df['user_id'].values
        self.item_ids = interactions_df['item_id'].values
        self.ratings = interactions_df['rating'].values

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        return (self.user_ids[idx], self.item_ids[idx], self.ratings[idx])

# Define the neural collaborative filtering model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=32):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        concatenated = torch.cat([user_embeds, item_embeds], dim=-1)
        output = self.mlp(concatenated)
        return output.squeeze()

# Instantiate the dataset and data loader
interaction_dataset = InteractionDataset(interactions_df)
data_loader = DataLoader(interaction_dataset, batch_size=32, shuffle=True)

# Instantiate the model, loss function, and optimizer
num_users = interactions_df['user_id'].nunique()
num_items = interactions_df['item_id'].nunique()
embedding_dim = 32

model = NCF(num_users, num_items, embedding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for user_ids, item_ids, ratings in data_loader:
        user_ids = user_ids.long()
        item_ids = item_ids.long()
        ratings = ratings.float()
        
        optimizer.zero_grad()
        outputs = model(user_ids, item_ids)
        loss = criterion(outputs, ratings)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(data_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')
