import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from datasetpfos import PFASDataset
from modelpfos import AttentionMIL
import config


def run_experiment(seed,
                   epochs=100,
                   lr=1e-4,
                   hidden_dim=64,
                   attention_dim=32,
                   weight_decay=1e-2,
                   dropout=0.0):


    # Seed固定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    all_pred = []
    all_true = []

    # Leave-One-Out
    for test_bag in range(1, 10):
        train_bags = [b for b in range(1, 10) if b != test_bag]

        train_dataset = PFASDataset(use_bags=train_bags)
        test_dataset = PFASDataset(
            use_bags=[test_bag],
            scaler=train_dataset.scaler,
            label_scaler=train_dataset.label_scaler
        )

        model = AttentionMIL(
            input_dim=config.INPUT_DIM,  # ← config の値を使用
            hidden_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )

        # L2正則化(weight_decay)を追加して過学習を強力に抑止
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

        criterion = nn.MSELoss()

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, generator=g
        )
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False
        )

        # ===== 学習 =====
        for epoch in range(epochs):
            model.train()
            for x, y in train_loader:
                optimizer.zero_grad()

                x = x.squeeze(0)
                y = y.squeeze(0).view(-1)

                pred = model(x).view(-1)
                loss = criterion(pred, y)

                loss.backward()
                optimizer.step()

        # ===== 評価 =====
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                x = x.squeeze(0)
                pred = model(x)

                pred = pred.detach().numpy().reshape(1, 1)
                pred = train_dataset.label_scaler.inverse_transform(pred)

                true = y.numpy().reshape(1, 1)
                true = train_dataset.label_scaler.inverse_transform(true)

                all_pred.append(pred[0, 0])
                all_true.append(true[0, 0])

    all_pred = np.array(all_pred)
    all_true = np.array(all_true)

    return {
        "MAE": mean_absolute_error(all_true, all_pred),
        "RMSE": np.sqrt(mean_squared_error(all_true, all_pred)),
        "R2": r2_score(all_true, all_pred)
    }


# ===== メイン実行ループ =====
if __name__ == "__main__":
    summary = []

    for hidden in config.HIDDEN_LIST:
        for attention in config.ATTENTION_LIST:
            for lr in config.LR_LIST:
                for epoch in config.EPOCH_LIST:
                    for wd in config.WEIGHT_DECAY_LIST:
                        for dropout in config.DROPOUT_LIST:

                            results = {"MAE": [], "RMSE": [], "R2": []}

                            for seed in config.SEEDS:
                                result = run_experiment(
                                    seed=seed,
                                    epochs=epoch,
                                    lr=lr,
                                    hidden_dim=hidden,
                                    attention_dim=attention,
                                    weight_decay=wd,
                                    dropout=dropout
                                )
                          
                                results["MAE"].append(result["MAE"])
                                results["RMSE"].append(result["RMSE"])
                                results["R2"].append(result["R2"])

                            score = np.mean(results["R2"]) - np.std(results["R2"])

                            summary.append({
                                "Hidden": hidden,
                                "Attention": attention,
                                "LearningRate": lr,
                                "Epoch": epoch,
                                "WeightDecay": wd,
                                "Dropout": dropout,
                                "MAE": np.mean(results["MAE"]),
                                "MAE_SD": np.std(results["MAE"]),
                                "RMSE": np.mean(results["RMSE"]),
                                "RMSE_SD": np.std(results["RMSE"]),
                                "R2": np.mean(results["R2"]),
                                "R2_SD": np.std(results["R2"]),
                                "Score": score,
                            })

                            print("=" * 60)
                            print(f"Hidden Dim    : {hidden}")
                            print(f"Attention Dim : {attention}")
                            print(f"Learning Rate : {lr:.0e}")
                            print(f"Epochs        : {epoch}")
                            print(f"Weight Decay : {wd:.0e}")
                            print(f"Dropout      : {dropout}")
                            print(f"Seeds         : {len(config.SEEDS)}")
                            print(f"\nPFOS")
                            print(f"  MAE   : {np.mean(results['MAE']):.2f} sd {np.std(results['MAE']):.2f}")
                            print(f"  RMSE  : {np.mean(results['RMSE']):.2f} sd {np.std(results['RMSE']):.2f}")
                            print(f"  R²    : {np.mean(results['R2']):.3f} sd {np.std(results['R2']):.3f}")
                            print(f"  Score : {score:.3f}")

    df = pd.DataFrame(summary)
    df.to_csv("hyperparameter_results_PFOS.csv", index=False, encoding="utf-8-sig")
    print("\nSaved : hyperparameter_results_PFOS.csv")
