import random

import numpy as np
import optuna
import torch
from torch import nn
from torchmetrics import Recall, Specificity

from src.classifier import PostClassifier

# Reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


class Trainer:
    def __init__(self, epochs=200, num_classes=3):
        self.epochs = epochs
        self.num_classes = num_classes
        self.metrics = {"recall": Recall(task="multiclass", average='micro', num_classes=self.num_classes),
                        "specificity": Specificity(task="multiclass", average='micro', num_classes=self.num_classes)}
        self.model = PostClassifier(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, lr, train_loader, test_loader, trial=None, logging=True):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        train_metrics, test_metrics = [], []

        for epoch in range(self.epochs):
            if logging:
                print(f"\nEpoch {epoch + 1}\n-------------------------------")
            train_metrics.append(
                self.batch_training(self.model, train_loader, optimizer, logging=logging))
            test_metrics.append(self.test(self.model, test_loader, self.metrics, logging=logging))
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

        for idx, (paragraph_embedding, label, _) in enumerate(train_loader):
            optimizer.zero_grad()
            predicted_label = model(paragraph_embedding)
            loss = self.criterion(predicted_label, label)
            loss.backward()
            optimizer.step()

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
