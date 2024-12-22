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

