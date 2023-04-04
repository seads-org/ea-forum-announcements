import argparse
import os
import random
from datetime import date

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Recall, Specificity

from src.classifier import PostClassifier
from src.data_utils import load_data
from src.paths import datap, modelp

# Reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class Trainer:
    def __init__(self, epochs=200, num_classes=3):
        self.epochs = epochs
        self.num_classes = num_classes
        self.metrics = {"recall": Recall(task="multiclass", average='macro', num_classes=self.num_classes),
                        "specificity": Specificity(task="multiclass", average='macro', num_classes=self.num_classes)}
        self.model = PostClassifier(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, lr, train_loader, test_loader, trial=None, logging=True):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_metrics, test_metrics = [], []
        for epoch in range(self.epochs):
            if logging:
                print(f"\nEpoch {epoch + 1}\n-------------------------------")
            train_metrics.append(self.batch_training(self.model,
                                                     train_loader,
                                                     optimizer,
                                                     logging=logging))
            test_metrics.append(self.test(self.model,
                                          test_loader,
                                          self.metrics,
                                          logging=logging))
            if trial is not None:
                test_recall = test_metrics[-1][1]
                trial.report(test_recall, epoch)
                # Prune unpromising trials if used for hyperparameter tuning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return train_metrics, test_metrics

    def batch_training(self, model, train_loader, optimizer, logging=True):
        model.train()
        for metric in self.metrics.values():
            metric.reset()

        for idx, (embedding, label, _) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_label = model(embedding)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            optimizer.step()

            for metric in self.metrics.values():
                metric(predicted_label, label)

        metrics_vals = {name: metric.compute().item() for name, metric in self.metrics.items()}
        if logging:
            print(f"train FPR {(1 - metrics_vals['specificity']):.3f} | train recall {metrics_vals['recall']:.3f}")
        return list(metrics_vals.values())

    def test(self, model, test_loader, metrics, logging=True):
        model.eval()
        for metric in metrics.values():
            metric.reset()

        with torch.no_grad():
            for idx, (paragraph_embedding, label, _) in enumerate(test_loader):
                predicted_label = model(paragraph_embedding)
                for metric in metrics.values():
                    metric(predicted_label, label)

        metrics_vals = {name: metric.compute().item() for name, metric in metrics.items()}
        if logging:
            print(f"test FPR {(1 - metrics_vals['specificity']):.3f} | test recall {metrics_vals['recall']:.3f}")
        return list(metrics_vals.values())

    @staticmethod
    def weight_reset(layer):
        if isinstance(layer, nn.Linear):
            layer.reset_parameters()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-test-split", default=True)
    parser.add_argument("--save-model", default=True)
    parser.add_argument("--model-name", default="", help="Additional details to include in model's name, "
                                                         "default: ea_embeddings_classifier:{date}.pth")

    args = parser.parse_args()
    dataset_size = len(pd.read_csv(datap("labeled_posts_embedded.csv")))
    trainer = Trainer()

    lr, batch_size = 0.005430718671093421, 64
    train_dataset = test_dataset = load_data()
    if args.train_test_split:
        train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

    print(len(train_dataset))
    print(len(test_dataset))
    print("Classes in the test set: ", torch.bincount(train_dataset.dataset.y[train_dataset.indices]))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_metrics, test_metrics = trainer.train(lr, train_loader, test_loader)

    torch.save(trainer.model,
               os.path.join(modelp(), f"ea_embeddings_classifier:{str(date.today())}:{args.model_name}.pth"))
