from functools import reduce
from operator import mul

from datasetpfos import PFASDataset
from run_pfos_nested_production import coarse_grid, fine_master_grid
from trainpfos_nested import run_nested_lobo


OUTPUT_DIR = "nested_production_PFOS_rain_1_7_30"
INNER_SEEDS = list(range(5))
OUTER_SEEDS = list(range(20))
OUTER_TEST_BAGS = list(range(1, 10))
REFINE_KEYS = ["Hidden", "LearningRate", "Epoch"]
N_JOBS = 4


def main():
    dataset = PFASDataset(use_bags=OUTER_TEST_BAGS)
    stage1_candidates = reduce(
        mul,
        (len(values) for values in coarse_grid.values()),
        1,
    )

    print(f"使用特徴量 ({dataset.input_dim}): {dataset.all_feature_cols}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")
    print(f"Inner seeds: {INNER_SEEDS}")
    print(f"Outer seeds: {OUTER_SEEDS}")
    print(f"n_jobs: {N_JOBS}")
    print(f"Stage 1候補数: {stage1_candidates}")
    print(
        "Stage 2: Stage 1上位3条件の周辺を探索 "
        f"(refine_keys={REFINE_KEYS}, 重複候補は除外)"
    )

    run_nested_lobo(
        coarse_grid=coarse_grid,
        fine_master_grid=fine_master_grid,
        refine_top_k=3,
        refine_keys=REFINE_KEYS,
        inner_seeds=INNER_SEEDS,
        outer_seeds=OUTER_SEEDS,
        outer_test_bags=OUTER_TEST_BAGS,
        output_dir=OUTPUT_DIR,
        n_jobs=N_JOBS,
        resume=True,
    )


if __name__ == "__main__":
    main()
