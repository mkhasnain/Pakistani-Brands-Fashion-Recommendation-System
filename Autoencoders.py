import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class InteractionDataset(Dataset):
    def __init__(self, interactions_df, num_users, num_items):
        self.user_item_matrix = interactions_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0).values
        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return torch.FloatTensor(self.user_item_matrix[idx])

interaction_dataset = InteractionDataset(interactions_df, num_users, num_items)
data_loader = DataLoader(interaction_dataset, batch_size=32, shuffle=True)

class Autoencoder(nn.Module):
    def __init__(self, num_items, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_items, 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_items),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

autoencoder = Autoencoder(num_items, encoding_dim=32)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)

num_epochs = 20

for epoch in range(num_epochs):
    autoencoder.train()
    total_loss = 0
    for user_ratings in data_loader:
        optimizer.zero_grad()
        outputs = autoencoder(user_ratings)
        loss = criterion(outputs, user_ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')


def recommend_for_user_autoencoder(user_id, num_recommendations=5):
    autoencoder.eval()
    user_ratings = torch.FloatTensor(interaction_dataset.user_item_matrix[user_id]).unsqueeze(0)
    
    with torch.no_grad():
        decoded_ratings = autoencoder(user_ratings)
    
    decoded_ratings = decoded_ratings.numpy().flatten()
    top_item_indices = decoded_ratings.argsort()[-num_recommendations:][::-1]
    recommended_items = df.iloc[top_item_indices].copy()
    
    # Inverse transform the categorical columns
    for column in label_encoders:
        recommended_items[column] = label_encoders[column].inverse_transform(recommended_items[column])
    
    return recommended_items

# Example usage: Get 5 recommendations for user 0
recommendations = recommend_for_user_autoencoder(0, 5)
print(recommendations)

