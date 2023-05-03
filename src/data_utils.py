import pandas as pd
import torch
from torch.utils.data import Dataset

from src.paths import datap


def load_data(dataset_filename="labeled_posts_embedded.csv"):
    embed_path = datap(dataset_filename)
    return Posts(embed_path)


class Posts(Dataset):

    def __init__(self, embed_path):
        embeddings_df = pd.read_csv(embed_path)

        self.x = torch.tensor(embeddings_df.drop(["postId", "label"], axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(embeddings_df["label"].values, dtype=torch.int64)
        self.ids = embeddings_df.postId

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.ids.iloc[idx]
