import os
import random
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

import config
from datasetpfos import PFASDataset
from modelpfos import AttentionMIL


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_metrics(true_values, pred_values):
    true_values = np.asarray(true_values, dtype=float)
    pred_values = np.asarray(pred_values, dtype=float)

    return {
        "MAE": mean_absolute_error(true_values, pred_values),
        "RMSE": np.sqrt(mean_squared_error(true_values, pred_values)),
        "R2": r2_score(true_values, pred_values),
    }


def train_and_predict(
    train_bags,
    eval_bag,
    seed,
    hidden_dim,
    attention_dim,
    lr,
    epochs,
    weight_decay,
    dropout,
):
    """Train on train_bags and return every PFOS observation in eval_bag."""
    assert eval_bag not in train_bags

    set_seed(seed)

    # Scalers are fitted only on the training Bags.
    train_dataset = PFASDataset(use_bags=train_bags)
    eval_dataset = PFASDataset(
        use_bags=[eval_bag],
        scaler=train_dataset.scaler,
        label_scaler=train_dataset.label_scaler,
    )

    model = AttentionMIL(
        input_dim=config.INPUT_DIM,
        hidden_dim=hidden_dim,
        attention_dim=attention_dim,
        dropout=dropout,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        generator=generator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=1,
        shuffle=False,
    )

    for _ in range(epochs):
        model.train()

        for x, y in train_loader:
            optimizer.zero_grad()

            x = x.squeeze(0)
            y = y.squeeze(0).view(-1)

            pred = model(x).view(-1)
            loss = criterion(pred, y)

            loss.backward()
            optimizer.step()

    predictions = []
    model.eval()

    with torch.no_grad():
        for sample_index, (x, y) in enumerate(eval_loader):
            x = x.squeeze(0)
            pred = model(x)

            pred = pred.detach().cpu().numpy().reshape(1, 1)
            pred = train_dataset.label_scaler.inverse_transform(pred)[0, 0]

            true = y.cpu().numpy().reshape(1, 1)
            true = train_dataset.label_scaler.inverse_transform(true)[0, 0]

            metadata = eval_dataset.label_df.iloc[sample_index]
            predictions.append(
                {
                    "SampleID": metadata["id"],
                    "Date": metadata["date"],
                    "River": metadata["river"],
                    "Location": metadata["location"],
                    "Pred": float(pred),
                    "True": float(true),
                }
            )

    assert len(predictions) == len(eval_dataset.label_df)
    assert [row["SampleID"] for row in predictions] == eval_dataset.label_df[
        "id"
    ].tolist()

    return predictions


def evaluate_inner_condition(
    outer_test_bag,
    outer_train_bags,
    params,
    inner_seeds,
):
    seed_metrics = []

    for seed in inner_seeds:
        seed_predictions = []

        for inner_val_bag in outer_train_bags:
            inner_train_bags = [
                bag for bag in outer_train_bags if bag != inner_val_bag
            ]

            assert outer_test_bag not in outer_train_bags
            assert outer_test_bag not in inner_train_bags
            assert outer_test_bag != inner_val_bag
            assert inner_val_bag not in inner_train_bags

            fold_predictions = train_and_predict(
                train_bags=inner_train_bags,
                eval_bag=inner_val_bag,
                seed=seed,
                hidden_dim=params["Hidden"],
                attention_dim=params["Attention"],
                lr=params["LearningRate"],
                epochs=params["Epoch"],
                weight_decay=params["WeightDecay"],
                dropout=params["Dropout"],
            )
            seed_predictions.extend(fold_predictions)

        metrics = calculate_metrics(
            [row["True"] for row in seed_predictions],
            [row["Pred"] for row in seed_predictions],
        )
        seed_metrics.append(metrics)

    r2_values = [metrics["R2"] for metrics in seed_metrics]

    return {
        "Inner_MAE_Mean": np.mean(
            [metrics["MAE"] for metrics in seed_metrics]
        ),
        "Inner_MAE_SD": np.std(
            [metrics["MAE"] for metrics in seed_metrics]
        ),
        "Inner_RMSE_Mean": np.mean(
            [metrics["RMSE"] for metrics in seed_metrics]
        ),
        "Inner_RMSE_SD": np.std(
            [metrics["RMSE"] for metrics in seed_metrics]
        ),
        "Inner_R2_Mean": np.mean(r2_values),
        "Inner_R2_SD": np.std(r2_values),
        "InnerCVScore": np.mean(r2_values) - np.std(r2_values),
    }


def select_best_params(
    outer_test_bag,
    outer_train_bags,
    hidden_list,
    attention_list,
    lr_list,
    epoch_list,
    weight_decay_list,
    dropout_list,
    inner_seeds,
):
    best_result = None
    all_results = []

    search_space = product(
        hidden_list,
        attention_list,
        lr_list,
        epoch_list,
        weight_decay_list,
        dropout_list,
    )

    for hidden, attention, lr, epoch, weight_decay, dropout in search_space:
        params = {
            "Hidden": hidden,
            "Attention": attention,
            "LearningRate": lr,
            "Epoch": epoch,
            "WeightDecay": weight_decay,
            "Dropout": dropout,
        }

        inner_result = evaluate_inner_condition(
            outer_test_bag=outer_test_bag,
            outer_train_bags=outer_train_bags,
            params=params,
            inner_seeds=inner_seeds,
        )
        candidate = {**params, **inner_result}
        all_results.append({"OuterTestBag": outer_test_bag, **candidate})

        if best_result is None:
            best_result = candidate
            continue

        candidate_key = (
            candidate["InnerCVScore"],
            candidate["Inner_R2_Mean"],
            -candidate["Inner_RMSE_Mean"],
            -candidate["Hidden"],
            -candidate["Attention"],
        )
        best_key = (
            best_result["InnerCVScore"],
            best_result["Inner_R2_Mean"],
            -best_result["Inner_RMSE_Mean"],
            -best_result["Hidden"],
            -best_result["Attention"],
        )

        if candidate_key > best_key:
            best_result = candidate

    return best_result, all_results


def save_results(
    outer_predictions,
    selected_params,
    inner_results,
    bag_metrics,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    predictions_df = pd.DataFrame(outer_predictions)
    selected_df = pd.DataFrame(selected_params)
    inner_results_df = pd.DataFrame(inner_results)
    bag_metrics_df = pd.DataFrame(bag_metrics)

    predictions_df.to_csv(
        os.path.join(output_dir, "nested_outer_predictions_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    selected_df.to_csv(
        os.path.join(output_dir, "nested_selected_params_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    inner_results_df.to_csv(
        os.path.join(output_dir, "nested_inner_results_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    bag_metrics_df.to_csv(
        os.path.join(output_dir, "nested_bag_metrics_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    seed_metric_rows = []
    for seed, seed_df in predictions_df.groupby("Seed", sort=True):
        metrics = calculate_metrics(seed_df["True"], seed_df["Pred"])
        seed_metric_rows.append(
            {
                "Seed": seed,
                "N_Samples": len(seed_df),
                **metrics,
            }
        )

    seed_metrics_df = pd.DataFrame(seed_metric_rows)
    assert seed_metrics_df["N_Samples"].nunique() == 1
    seed_metrics_df.to_csv(
        os.path.join(output_dir, "nested_seed_metrics_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    ensemble_df = (
        predictions_df.groupby(
            ["TestBag", "SampleID"],
            as_index=False,
            sort=True,
        )
        .agg(
            **{
                "Date": ("Date", "first"),
                "River": ("River", "first"),
                "Location": ("Location", "first"),
                "PredMean": ("Pred", "mean"),
                "PredSD": ("Pred", "std"),
                "True": ("True", "first"),
                "NSeeds": ("Seed", "nunique"),
            }
        )
    )

    true_counts = predictions_df.groupby(["TestBag", "SampleID"])["True"].nunique()
    assert (true_counts == 1).all()
    assert (ensemble_df["NSeeds"] == seed_metrics_df["Seed"].nunique()).all()

    ensemble_df.to_csv(
        os.path.join(output_dir, "nested_ensemble_predictions_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    ensemble_metrics = calculate_metrics(
        ensemble_df["True"],
        ensemble_df["PredMean"],
    )

    final_metrics_df = pd.DataFrame(
        [
            {
                "MAE_Mean": seed_metrics_df["MAE"].mean(),
                "MAE_SD": seed_metrics_df["MAE"].std(ddof=0),
                "RMSE_Mean": seed_metrics_df["RMSE"].mean(),
                "RMSE_SD": seed_metrics_df["RMSE"].std(ddof=0),
                "R2_Mean": seed_metrics_df["R2"].mean(),
                "R2_SD": seed_metrics_df["R2"].std(ddof=0),
                "N_Seeds": seed_metrics_df["Seed"].nunique(),
                "N_Samples_Per_Seed": seed_metrics_df["N_Samples"].min(),
                "Ensemble_MAE": ensemble_metrics["MAE"],
                "Ensemble_RMSE": ensemble_metrics["RMSE"],
                "Ensemble_R2": ensemble_metrics["R2"],
            }
        ]
    )
    final_metrics_df.to_csv(
        os.path.join(output_dir, "nested_final_metrics_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )


def run_nested_lobo(
    hidden_list=None,
    attention_list=None,
    lr_list=None,
    epoch_list=None,
    weight_decay_list=None,
    dropout_list=None,
    inner_seeds=None,
    outer_seeds=None,
    outer_test_bags=None,
    all_bags=None,
    output_dir=".",
):
    hidden_list = list(config.HIDDEN_LIST if hidden_list is None else hidden_list)
    attention_list = list(
        config.ATTENTION_LIST if attention_list is None else attention_list
    )
    lr_list = list(config.LR_LIST if lr_list is None else lr_list)
    epoch_list = list(config.EPOCH_LIST if epoch_list is None else epoch_list)
    weight_decay_list = list(
        config.WEIGHT_DECAY_LIST
        if weight_decay_list is None
        else weight_decay_list
    )
    dropout_list = list(
        config.DROPOUT_LIST if dropout_list is None else dropout_list
    )
    inner_seeds = list(config.SEEDS if inner_seeds is None else inner_seeds)
    outer_seeds = list(config.SEEDS if outer_seeds is None else outer_seeds)
    all_bags = list(config.TEST_BAGS if all_bags is None else all_bags)
    outer_test_bags = list(
        all_bags if outer_test_bags is None else outer_test_bags
    )

    assert hidden_list
    assert attention_list
    assert lr_list
    assert epoch_list
    assert weight_decay_list
    assert dropout_list
    assert inner_seeds
    assert outer_seeds
    assert outer_test_bags
    assert set(outer_test_bags).issubset(set(all_bags))

    outer_predictions = []
    selected_params = []
    inner_results = []
    bag_metrics = []

    for outer_test_bag in outer_test_bags:
        outer_train_bags = [bag for bag in all_bags if bag != outer_test_bag]
        assert outer_test_bag not in outer_train_bags

        print(f"\nOuter Test Bag: {outer_test_bag}")
        print("Inner探索開始")

        best, outer_inner_results = select_best_params(
            outer_test_bag=outer_test_bag,
            outer_train_bags=outer_train_bags,
            hidden_list=hidden_list,
            attention_list=attention_list,
            lr_list=lr_list,
            epoch_list=epoch_list,
            weight_decay_list=weight_decay_list,
            dropout_list=dropout_list,
            inner_seeds=inner_seeds,
        )
        inner_results.extend(outer_inner_results)

        selected_row = {"TestBag": outer_test_bag, **best}
        selected_params.append(selected_row)

        print(
            "選択された最良条件: "
            f"Hidden={best['Hidden']}, "
            f"Attention={best['Attention']}, "
            f"LearningRate={best['LearningRate']}, "
            f"Epoch={best['Epoch']}, "
            f"WeightDecay={best['WeightDecay']}, "
            f"Dropout={best['Dropout']}, "
            f"InnerCVScore={best['InnerCVScore']:.6f}"
        )

        for seed in outer_seeds:
            seed_predictions = train_and_predict(
                train_bags=outer_train_bags,
                eval_bag=outer_test_bag,
                seed=seed,
                hidden_dim=best["Hidden"],
                attention_dim=best["Attention"],
                lr=best["LearningRate"],
                epochs=best["Epoch"],
                weight_decay=best["WeightDecay"],
                dropout=best["Dropout"],
            )

            for row in seed_predictions:
                outer_predictions.append(
                    {
                        "TestBag": outer_test_bag,
                        **row,
                        "Seed": seed,
                        "Hidden": best["Hidden"],
                        "Attention": best["Attention"],
                        "LearningRate": best["LearningRate"],
                        "Epoch": best["Epoch"],
                        "WeightDecay": best["WeightDecay"],
                        "Dropout": best["Dropout"],
                        "InnerCVScore": best["InnerCVScore"],
                    }
                )

            metrics = calculate_metrics(
                [row["True"] for row in seed_predictions],
                [row["Pred"] for row in seed_predictions],
            )
            bag_metrics.append(
                {
                    "TestBag": outer_test_bag,
                    "Seed": seed,
                    "N_Samples": len(seed_predictions),
                    "MAE": metrics["MAE"],
                    "RMSE": metrics["RMSE"],
                }
            )

        print(
            f"Outer予測完了: Bag={outer_test_bag}, "
            f"Samples={len(seed_predictions)}, Seeds={len(outer_seeds)}"
        )

    save_results(
        outer_predictions=outer_predictions,
        selected_params=selected_params,
        inner_results=inner_results,
        bag_metrics=bag_metrics,
        output_dir=output_dir,
    )

    print("\nNested LOBO完了")
    print(f"Results saved in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    run_nested_lobo()
