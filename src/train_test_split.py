import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import datap

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-fraction", default=0.9)
    args = parser.parse_args()

    dataset_filename = "labeled_posts_embedded.csv"
    embed_path = datap(dataset_filename)
    dataset_size = len(pd.read_csv(embed_path))

    train_indices, test_indices = train_test_split(np.arange(dataset_size),
                                                   train_size=args.train_fraction,
                                                   random_state=42)

    np.savetxt(datap("train_indices.csv"), train_indices, fmt='%i', delimiter=",")
    np.savetxt(datap("test_indices.csv"), test_indices, fmt='%i', delimiter=",")
