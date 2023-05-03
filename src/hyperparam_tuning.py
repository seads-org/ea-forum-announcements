import argparse
import warnings
from datetime import date

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler, DataLoader

from src.data_utils import load_data
from src.paths import datap, modelp
from src.training import seed, Trainer


class Tuner:
    def __init__(self, trainer, dataset_size, k_folds=3):
        self.trainer = trainer
        self.dataset_size = dataset_size
        self.k_folds = k_folds

    def cross_validation(self, trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [1, 8, 16, 32, 64, self.dataset_size])

        logging = False
        folds_metrics = np.zeros(self.k_folds)
        dataset = load_data()
        splits = KFold(n_splits=self.k_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(val_idx)
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

            self.trainer.train(lr, train_loader, test_loader, trial=trial, logging=logging)
            test_specificity, test_recall = self.trainer.test(self.trainer.model, test_loader, logging=logging)
            folds_metrics[fold] = (test_recall + test_specificity) / 2

        if trial is not None:
            self.trainer.model.apply(self.trainer.weight_reset)
        return np.mean(folds_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Hyperparameter tuning script',
        description='Hyperparameter tuning script based on cross validation where objective '
                    'is to maximize averaged recall and specificity')

    parser.add_argument('--filename', default="labeled_posts_embedded.csv",
                        help="Filename of the .csv file with embeddings for posts")
    parser.add_argument('--model-name', default="ea_embeddings_post_classifier", help="Filename for model saving")
    parser.add_argument('--n-trials', default=50, help="Number of trials for Optuna to run")

    args = parser.parse_args()

    dataset_size = len(pd.read_csv(datap(args.filename)))
    tuner = Tuner(Trainer(), dataset_size)

    model_name = f"{args.model_name}:{str(date.today())}"
    study = optuna.create_study(direction="maximize",
                                storage="sqlite:///db.sqlite3",
                                study_name=model_name,
                                load_if_exists=True)

    # With cross validation there are warnings about the value of objective
    # that was already calculated for one epoch, we don't care,
    # we assume pruning based on info from one fold is enough
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
    study.optimize(tuner.cross_validation, n_trials=args.n_trials)

    trial = study.best_trial

    print("The best objective: {}".format(trial.value))
    print("The best hyperparameters: {}".format(trial.params))

    # Retrain the best model on the whole dataset
    trainer = Trainer(epochs=10, per_paragraph=True)
    lr, batch_size = trial.params["lr"], trial.params["batch_size"]
    dataset = load_data(args.filename)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    train_metrics, test_metrics = trainer.train(lr, data_loader, data_loader)

    torch.save(trainer.model, modelp(f"{model_name}.pth"))
