import argparse
import logging
import os
import random
from collections import defaultdict, Counter
from datetime import date
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchmetrics import Recall, Specificity

from src.classifier import PostClassifier
from src.data_utils import load_data
from src.paths import datap, modelp, outputp

# Reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

log = logging.getLogger(__name__)


class Trainer:
    def __init__(self, epochs=200, num_classes=3, per_paragraph=False):
        self.epochs = epochs
        self.num_classes = num_classes
        self.metrics = {"recall": Recall(task="multiclass", average='macro', num_classes=self.num_classes),
                        "specificity": Specificity(task="multiclass", average='macro', num_classes=self.num_classes)}
        self.model = PostClassifier(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.per_paragraph = per_paragraph

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
            test_metrics.append(self.test(self.model, test_loader, logging=logging))
            if trial is not None:
                test_metric = sum(test_metrics[-1]) / 2
                trial.report(test_metric, epoch)
                # Prune unpromising trials if used for hyperparameter tuning
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

        return train_metrics, test_metrics

    def batch_training(self, model, train_loader, optimizer, logging=True):
        model.train()
        for metric in self.metrics.values():
            metric.reset()

        preds_per_post = defaultdict(list)
        labels_per_post = {}
        for idx, (embeddings, labels, post_ids) in enumerate(train_loader):
            optimizer.zero_grad()
            preds = model(embeddings)
            loss = self.criterion(preds, labels)
            loss.backward()
            optimizer.step()

            predicted_labels = torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1)
            if not predicted_labels.size():
                predicted_labels = predicted_labels.unsqueeze(0)
            if self.per_paragraph:
                for idx, post_id in enumerate(post_ids):
                    preds_per_post[post_id].append(predicted_labels[idx].item())
                    labels_per_post[post_id] = labels[idx].item()
            else:
                for metric in self.metrics.values():
                    metric(predicted_labels, labels)

            if self.per_paragraph:
                preds_per_post = {k: Counter(v) for k, v in preds_per_post.items()}
                preds_per_post = {k: max(v, key=v.get) for k, v in preds_per_post.items()}
                for metric in self.metrics.values():
                    for post_id in preds_per_post:
                        predicted_label = preds_per_post[post_id]
                        if not predicted_label.size():
                            predicted_label = predicted_label.unsqueeze(0)
                        label = labels_per_post[post_id]
                        metric(predicted_label, label)

        metrics_vals = {name: metric.compute().item() for name, metric in self.metrics.items()}
        if logging:
            print(f"train FPR {(1 - metrics_vals['specificity']):.3f} | train recall {metrics_vals['recall']:.3f}")
        return list(metrics_vals.values())

    def test(self, model, test_loader, logging=True):
        model.eval()
        for metric in self.metrics.values():
            metric.reset()

        preds_per_post = defaultdict(list)
        labels_per_post = {}
        with torch.no_grad():
            for idx, (embeddings, labels, post_ids) in enumerate(test_loader):
                preds = model(embeddings)
                predicted_labels = torch.argmax(torch.nn.functional.softmax(preds, dim=1), dim=1)
                if not predicted_labels.size():
                    predicted_labels = predicted_labels.unsqueeze(0)
                if self.per_paragraph:
                    for idx, post_id in enumerate(post_ids):
                        preds_per_post[post_id].append(predicted_labels[idx])
                        labels_per_post[post_id] = labels[idx]
                else:
                    for metric in self.metrics.values():
                        metric(predicted_labels, labels)

            if self.per_paragraph:
                preds_per_post = {k: Counter(v) for k, v in preds_per_post.items()}
                preds_per_post = {k: max(v, key=v.get) for k, v in preds_per_post.items()}
                for metric in self.metrics.values():
                    for post_id in preds_per_post:
                        predicted_label = preds_per_post[post_id]
                        if not predicted_label.size():
                            predicted_label = predicted_label.unsqueeze(0)
                        label = labels_per_post[post_id]
                        metric(predicted_label, label)

        metrics_vals = {name: metric.compute().item() for name, metric in self.metrics.items()}
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
    parser.add_argument("--per-paragraph", action="store_true")
    parser.add_argument("--save-model", default=False)
    parser.add_argument('--filename', default="labeled_posts_embedded.csv",
                        help="Filename of the .csv file with embeddings for posts")
    parser.add_argument("--model-name", default="ea_embeddings_post_classifier", help="Filename for model saving")

    args = parser.parse_args()
    dataset_size = len(pd.read_csv(datap("labeled_paragraphs_embedded_e5base.csv")))
    trainer = Trainer(epochs=10, per_paragraph=args.per_paragraph)

    try:
        db_path = outputp("db.sqlite3")
        loaded_study = optuna.load_study(study_name=args.model_name,
                                         storage=f"sqlite:///{db_path}")
        log.info(f"Loading Optuna trail: {args.model_name}")
        lr, batch_size = loaded_study.best_trial.params["lr"], loaded_study.best_trial.params["batch_size"]
    except Exception:
        lr, batch_size = 0.01, 8
        log.warning(
            f"Couldn't find a matching Optuna study, falling back to default, learning rate: {lr}, batch_size: {batch_size}")

    dataset = load_data(args.filename)
    if args.train_test_split:
        train_indices = np.loadtxt(datap("train_indices.csv"))
        test_indices = np.loadtxt(datap("train_indices.csv"))
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)
    else:
        train_dataset = test_dataset = dataset

    log.info("Number of training examples: ", len(train_dataset))
    log.info("Number of testing examples: ", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    train_metrics, test_metrics = trainer.train(lr, train_loader, test_loader)

    if args.save_model:
        model_name = f"{args.model_name}:{str(date.today())}"
        torch.save(trainer.model, modelp(model_name))
