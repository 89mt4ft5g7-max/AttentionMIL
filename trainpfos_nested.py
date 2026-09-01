import os
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
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


PARAMETER_NAMES = (
    "Hidden",
    "Attention",
    "LearningRate",
    "Epoch",
    "WeightDecay",
    "Dropout",
)


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


def initialize_grid_worker():
    """Limit PyTorch thread pools inside each grid-search worker."""
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch only permits setting inter-op threads before parallel work
        # starts. In that case, keep the existing inter-op thread setting.
        pass


def evaluate_grid_candidate(
    outer_test_bag,
    outer_train_bags,
    params,
    inner_seeds,
):
    """Evaluate one hyperparameter candidate across all inner folds/seeds."""
    inner_result = evaluate_inner_condition(
        outer_test_bag=outer_test_bag,
        outer_train_bags=outer_train_bags,
        params=params,
        inner_seeds=inner_seeds,
    )
    candidate = {**params, **inner_result}
    return {"OuterTestBag": outer_test_bag, **candidate}


def evaluate_grid(
    outer_test_bag,
    outer_train_bags,
    candidates,
    inner_seeds,
    n_jobs=1,
):
    if n_jobs == 1:
        return [
            evaluate_grid_candidate(
                outer_test_bag=outer_test_bag,
                outer_train_bags=outer_train_bags,
                params=params,
                inner_seeds=inner_seeds,
            )
            for params in candidates
        ]

    results = [None] * len(candidates)
    with ProcessPoolExecutor(
        max_workers=n_jobs,
        initializer=initialize_grid_worker,
    ) as executor:
        future_to_index = {
            executor.submit(
                evaluate_grid_candidate,
                outer_test_bag,
                outer_train_bags,
                params,
                inner_seeds,
            ): index
            for index, params in enumerate(candidates)
        }
        for future in as_completed(future_to_index):
            results[future_to_index[future]] = future.result()

    return results


def result_sort_key(result):
    return (
        result["InnerCVScore"],
        result["Inner_R2_Mean"],
        -result["Inner_RMSE_Mean"],
        -result["Hidden"],
        -result["Attention"],
    )


def grid_candidates(grid):
    return [
        dict(zip(PARAMETER_NAMES, values))
        for values in product(*(grid[name] for name in PARAMETER_NAMES))
    ]


def neighboring_values(master_values, center):
    values = sorted(set(master_values))
    assert center in values

    center_index = values.index(center)
    lower_index = max(0, center_index - 1)
    upper_index = min(len(values), center_index + 2)
    return values[lower_index:upper_index]


def generate_fine_candidates(top_results, fine_master_grid, refine_keys):
    unique_candidates = {}
    refine_keys = set(refine_keys)

    for result in top_results:
        local_grid = {
            name: (
                neighboring_values(fine_master_grid[name], result[name])
                if name in refine_keys
                else [result[name]]
            )
            for name in PARAMETER_NAMES
        }

        for candidate in grid_candidates(local_grid):
            key = tuple(candidate[name] for name in PARAMETER_NAMES)
            unique_candidates[key] = candidate

    return list(unique_candidates.values())


def coarse_to_fine_search(
    outer_test_bag,
    outer_train_bags,
    coarse_grid,
    fine_master_grid,
    refine_top_k,
    refine_keys,
    inner_seeds,
    n_jobs,
):
    stage1_results = evaluate_grid(
        outer_test_bag=outer_test_bag,
        outer_train_bags=outer_train_bags,
        candidates=grid_candidates(coarse_grid),
        inner_seeds=inner_seeds,
        n_jobs=n_jobs,
    )
    ranked_stage1 = sorted(stage1_results, key=result_sort_key, reverse=True)
    top_results = ranked_stage1[: min(refine_top_k, len(ranked_stage1))]

    fine_candidates = generate_fine_candidates(
        top_results,
        fine_master_grid,
        refine_keys,
    )
    print(
        "Inner探索開始: Stage 2 Fine Grid Search "
        f"({len(fine_candidates)} candidates)"
    )
    stage2_results = evaluate_grid(
        outer_test_bag=outer_test_bag,
        outer_train_bags=outer_train_bags,
        candidates=fine_candidates,
        inner_seeds=inner_seeds,
        n_jobs=n_jobs,
    )
    ranked_stage2 = sorted(stage2_results, key=result_sort_key, reverse=True)
    best_result = {
        key: value
        for key, value in ranked_stage2[0].items()
        if key != "OuterTestBag"
    }

    return best_result, ranked_stage1, ranked_stage2


def save_results(
    outer_predictions,
    selected_params,
    stage1_results,
    stage2_results,
    bag_metrics,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    predictions_df = pd.DataFrame(outer_predictions)
    selected_df = pd.DataFrame(selected_params)
    stage1_results_df = pd.DataFrame(stage1_results)
    stage2_results_df = pd.DataFrame(stage2_results)
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
    stage2_results_df.to_csv(
        os.path.join(output_dir, "nested_inner_results_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    stage1_results_df.to_csv(
        os.path.join(output_dir, "nested_inner_stage1_results_PFOS.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    stage2_results_df.to_csv(
        os.path.join(output_dir, "nested_inner_stage2_results_PFOS.csv"),
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


def atomic_write_csv(rows, path):
    """Write a CSV completely before atomically replacing its destination."""
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        pd.DataFrame(rows).to_csv(
            temporary_path,
            index=False,
            encoding="utf-8-sig",
        )
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def atomic_write_json(data, path):
    """Write a JSON file completely before atomically replacing it."""
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def checkpoint_paths(output_dir, outer_test_bag):
    prefix = os.path.join(output_dir, f"checkpoint_bag_{outer_test_bag}")
    return {
        "stage1": f"{prefix}_stage1.csv",
        "stage2": f"{prefix}_stage2.csv",
        "selected": f"{prefix}_selected.csv",
        "predictions": f"{prefix}_predictions.csv",
        "bag_metrics": f"{prefix}_bag_metrics.csv",
        "complete": f"{prefix}_complete.json",
    }


def save_bag_checkpoint(
    outer_test_bag,
    stage1_results,
    stage2_results,
    selected_params,
    outer_predictions,
    bag_metrics,
    output_dir,
):
    """Persist one fully evaluated Outer Bag; create its marker last."""
    paths = checkpoint_paths(output_dir, outer_test_bag)
    checkpoint_rows = {
        "stage1": stage1_results,
        "stage2": stage2_results,
        "selected": selected_params,
        "predictions": outer_predictions,
        "bag_metrics": bag_metrics,
    }

    for name, rows in checkpoint_rows.items():
        atomic_write_csv(rows, paths[name])

    atomic_write_json(
        {
            "OuterTestBag": outer_test_bag,
            "files": {
                name: os.path.basename(paths[name])
                for name in checkpoint_rows
            },
            "row_counts": {
                name: len(rows) for name, rows in checkpoint_rows.items()
            },
        },
        paths["complete"],
    )


def load_bag_checkpoint(output_dir, outer_test_bag):
    """Load a Bag only when its last-written completion marker is valid."""
    paths = checkpoint_paths(output_dir, outer_test_bag)
    if not os.path.isfile(paths["complete"]):
        return None

    try:
        with open(paths["complete"], encoding="utf-8") as file:
            marker = json.load(file)
        if marker.get("OuterTestBag") != outer_test_bag:
            return None

        loaded = {}
        for name in (
            "stage1",
            "stage2",
            "selected",
            "predictions",
            "bag_metrics",
        ):
            if not os.path.isfile(paths[name]):
                return None
            frame = pd.read_csv(paths[name])
            if len(frame) != marker.get("row_counts", {}).get(name):
                return None
            loaded[name] = frame.to_dict("records")
    except (OSError, ValueError, pd.errors.ParserError):
        return None

    return loaded


def invalidate_bag_checkpoint(output_dir, outer_test_bag):
    """Remove only the completion marker before recalculating a Bag."""
    complete_path = checkpoint_paths(output_dir, outer_test_bag)["complete"]
    if os.path.exists(complete_path):
        os.remove(complete_path)


def default_coarse_grid():
    return {
        "Hidden": config.HIDDEN_LIST,
        "Attention": config.ATTENTION_LIST,
        "LearningRate": config.LR_LIST,
        "Epoch": config.EPOCH_LIST,
        "WeightDecay": config.WEIGHT_DECAY_LIST,
        "Dropout": config.DROPOUT_LIST,
    }


def normalize_grid(grid):
    normalized = {name: list(grid[name]) for name in PARAMETER_NAMES}
    assert all(normalized[name] for name in PARAMETER_NAMES)
    return normalized


def run_nested_lobo(
    coarse_grid=None,
    fine_master_grid=None,
    refine_top_k=3,
    refine_keys=None,
    inner_seeds=None,
    outer_seeds=None,
    outer_test_bags=None,
    all_bags=None,
    output_dir=".",
    n_jobs=1,
    resume=False,
):
    os.makedirs(output_dir, exist_ok=True)

    coarse_grid = normalize_grid(
        default_coarse_grid() if coarse_grid is None else coarse_grid
    )
    fine_master_grid = normalize_grid(
        coarse_grid if fine_master_grid is None else fine_master_grid
    )
    refine_keys = list(PARAMETER_NAMES if refine_keys is None else refine_keys)
    invalid_refine_keys = [
        name for name in refine_keys if name not in PARAMETER_NAMES
    ]
    if invalid_refine_keys:
        raise ValueError(
            "Invalid refine_keys: "
            f"{invalid_refine_keys}. Allowed keys: {list(PARAMETER_NAMES)}"
        )
    inner_seeds = list(config.SEEDS if inner_seeds is None else inner_seeds)
    outer_seeds = list(config.SEEDS if outer_seeds is None else outer_seeds)
    all_bags = list(config.TEST_BAGS if all_bags is None else all_bags)
    outer_test_bags = list(
        all_bags if outer_test_bags is None else outer_test_bags
    )

    assert refine_top_k >= 1
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    for name in PARAMETER_NAMES:
        assert set(coarse_grid[name]).issubset(set(fine_master_grid[name]))
    assert inner_seeds
    assert outer_seeds
    assert outer_test_bags
    assert set(outer_test_bags).issubset(set(all_bags))

    outer_predictions = []
    selected_params = []
    stage1_results = []
    stage2_results = []
    bag_metrics = []

    for outer_test_bag in outer_test_bags:
        if resume:
            checkpoint = load_bag_checkpoint(output_dir, outer_test_bag)
            if checkpoint is not None:
                stage1_results.extend(checkpoint["stage1"])
                stage2_results.extend(checkpoint["stage2"])
                selected_params.extend(checkpoint["selected"])
                outer_predictions.extend(checkpoint["predictions"])
                bag_metrics.extend(checkpoint["bag_metrics"])
                print(
                    f"\nOuter Test Bag: {outer_test_bag} "
                    "(checkpoint完了済みのためskip)"
                )
                continue

        invalidate_bag_checkpoint(output_dir, outer_test_bag)

        outer_train_bags = [bag for bag in all_bags if bag != outer_test_bag]
        assert outer_test_bag not in outer_train_bags

        print(f"\nOuter Test Bag: {outer_test_bag}")
        print("Inner探索開始: Stage 1 Coarse Grid Search")

        best, outer_stage1_results, outer_stage2_results = coarse_to_fine_search(
            outer_test_bag=outer_test_bag,
            outer_train_bags=outer_train_bags,
            coarse_grid=coarse_grid,
            fine_master_grid=fine_master_grid,
            refine_top_k=refine_top_k,
            refine_keys=refine_keys,
            inner_seeds=inner_seeds,
            n_jobs=n_jobs,
        )
        selected_row = {"TestBag": outer_test_bag, **best}
        outer_selected_params = [selected_row]
        outer_predictions_for_bag = []
        outer_bag_metrics = []

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
                outer_predictions_for_bag.append(
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
            outer_bag_metrics.append(
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

        save_bag_checkpoint(
            outer_test_bag=outer_test_bag,
            stage1_results=outer_stage1_results,
            stage2_results=outer_stage2_results,
            selected_params=outer_selected_params,
            outer_predictions=outer_predictions_for_bag,
            bag_metrics=outer_bag_metrics,
            output_dir=output_dir,
        )

        stage1_results.extend(outer_stage1_results)
        stage2_results.extend(outer_stage2_results)
        selected_params.extend(outer_selected_params)
        outer_predictions.extend(outer_predictions_for_bag)
        bag_metrics.extend(outer_bag_metrics)

    save_results(
        outer_predictions=outer_predictions,
        selected_params=selected_params,
        stage1_results=stage1_results,
        stage2_results=stage2_results,
        bag_metrics=bag_metrics,
        output_dir=output_dir,
    )

    print("\nNested LOBO完了")
    print(f"Results saved in: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    run_nested_lobo()
