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

