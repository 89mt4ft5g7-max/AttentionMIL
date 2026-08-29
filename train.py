import os
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from dataset import PFASDataset
from model import AttentionMIL
import config

def run_experiment(seed,
                   epochs=100,
                   lr=1e-4,
                   hidden_dim=64,
                   attention_dim=32
):

    # -------------------
    # Seed固定
    # -------------------
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_pred = []
    all_true = []

    # Leave-One-Out
    for test_bag in range(1, 10):

        train_bags = [
            b for b in range(1, 10)
            if b != test_bag
        ]

        train_dataset = PFASDataset(use_bags=train_bags)

        test_dataset = PFASDataset(
            use_bags=[test_bag],
            scaler=train_dataset.scaler,
            label_scaler=train_dataset.label_scaler
        )

        model = AttentionMIL(
            input_dim=15,
            hidden_dim=hidden_dim,
            attention_dim=attention_dim
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr
        )

        criterion = nn.MSELoss()

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            generator=g
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False
        )

        # ===== 学習 =====

        for epoch in range(epochs):

            model.train()

            for x, y in train_loader:

                optimizer.zero_grad()

                x = x.squeeze(0)
                y = y.squeeze(0)

                pred = model(x).squeeze()

                loss = criterion(pred, y)

                loss.backward()

                optimizer.step()

        # ===== 評価 =====

        model.eval()

        with torch.no_grad():

            for x, y in test_loader:

                x = x.squeeze(0)

                pred = model(x)

                pred = pred.detach().numpy().reshape(1,3)
                pred = train_dataset.label_scaler.inverse_transform(pred)

                true = y.numpy().reshape(1,3)
                true = train_dataset.label_scaler.inverse_transform(true)

                all_pred.append(pred)
                all_true.append(true)

    all_pred = np.vstack(all_pred)
    all_true = np.vstack(all_true)

    result = {}

    names = ["PFOS","PFOA","Total"]

    for i,name in enumerate(names):

        result[name] = {
            "MAE":
                mean_absolute_error(
                    all_true[:,i],
                    all_pred[:,i]
                ),

            "RMSE":
                np.sqrt(
                    mean_squared_error(
                        all_true[:,i],
                        all_pred[:,i]
                    )
                ),

            "R2":
                r2_score(
                    all_true[:,i],
                    all_pred[:,i]
                )
        }

    return result


import numpy as np
import config

summary = []

for hidden in config.HIDDEN_LIST:
    for attention in config.ATTENTION_LIST:
        for lr in config.LR_LIST:
            for epoch in config.EPOCH_LIST:

                results = {
                    "PFOS": {"MAE": [], "RMSE": [], "R2": []},
                    "PFOA": {"MAE": [], "RMSE": [], "R2": []},
                    "Total": {"MAE": [], "RMSE": [], "R2": []},
                }

                for seed in config.SEEDS:
    
                    result = run_experiment(
                        seed=seed,
                        epochs=epoch,
                        lr=lr,
                        hidden_dim=hidden,
                        attention_dim=attention
                    )

                    for name in ["PFOS", "PFOA", "Total"]:
                        results[name]["MAE"].append(result[name]["MAE"])
                        results[name]["RMSE"].append(result[name]["RMSE"])
                        results[name]["R2"].append(result[name]["R2"])
  
                summary.append({
                    "Hidden": hidden,
                    "Attention": attention,
                    "LearningRate": lr,
                    "Epoch": epoch,

                    "PFOS_MAE": np.mean(results["PFOS"]["MAE"]),
                    "PFOS_MAE_SD": np.std(results["PFOS"]["MAE"]),
                    "PFOS_RMSE": np.mean(results["PFOS"]["RMSE"]),
                    "PFOS_RMSE_SD": np.std(results["PFOS"]["RMSE"]),
                    "PFOS_R2": np.mean(results["PFOS"]["R2"]),
                    "PFOS_R2_SD": np.std(results["PFOS"]["R2"]),

                    "PFOA_MAE": np.mean(results["PFOA"]["MAE"]),
                    "PFOA_MAE_SD": np.std(results["PFOA"]["MAE"]),
                    "PFOA_RMSE": np.mean(results["PFOA"]["RMSE"]),
                    "PFOA_RMSE_SD": np.std(results["PFOA"]["RMSE"]),
                    "PFOA_R2": np.mean(results["PFOA"]["R2"]),
                    "PFOA_R2_SD": np.std(results["PFOA"]["R2"]),

                    "Total_MAE": np.mean(results["Total"]["MAE"]),
                    "Total_MAE_SD": np.std(results["Total"]["MAE"]),
                    "Total_RMSE": np.mean(results["Total"]["RMSE"]),
                    "Total_RMSE_SD": np.std(results["Total"]["RMSE"]),
                    "Total_R2": np.mean(results["Total"]["R2"]),
                    "Total_R2_SD": np.std(results["Total"]["R2"]),
                })

                print("=" * 60)
                print(f"Hidden Dim    : {hidden}")
                print(f"Attention Dim : {attention}")
                print(f"Learning Rate : {lr:.0e}")
                print(f"Epochs        : {epoch}")
                print(f"Seeds         : {len(config.SEEDS)}")

                for name in ["PFOS", "PFOA", "Total"]:
                    print(f"\n{name}")
                    print(f"  MAE  : {np.mean(results[name]['MAE']):.2f} sd {np.std(results[name]['MAE']):.2f}")
                    print(f"  RMSE : {np.mean(results[name]['RMSE']):.2f} sd {np.std(results[name]['RMSE']):.2f}")
                    print(f"  R²   : {np.mean(results[name]['R2']):.3f} sd {np.std(results[name]['R2']):.3f}")


df = pd.DataFrame(summary)

df.to_csv(
    "hyperparameter_results.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved : hyperparameter_results.csv")