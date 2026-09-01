from trainpfos_nested import run_nested_lobo


coarse_grid = {
    "Hidden": [8, 24, 64],
    "Attention": [2, 4, 12, 16],
    "LearningRate": [1e-3, 3e-3],
    "Epoch": [100, 150],
    "WeightDecay": [0, 1e-4],
    "Dropout": [0.0, 0.1],
}


fine_master_grid = {
    "Hidden": [
        8, 12, 16, 20, 24, 32, 40, 48, 64
    ],
    "Attention": [
        2, 4, 8, 12, 16
    ],
    "LearningRate": [
        5e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2, 2e-2
    ],
    "Epoch": [
        50, 75, 100, 125, 150, 200, 250
    ],
    "WeightDecay": [
        0, 1e-5, 1e-4, 1e-3
    ],
    "Dropout": [
        0.0, 0.1, 0.2
    ],
}


def main():
    run_nested_lobo(
        coarse_grid=coarse_grid,
        fine_master_grid=fine_master_grid,

        refine_top_k=3,

        refine_keys=[
            "Hidden",
            "LearningRate",
            "Epoch",
        ],

        inner_seeds=list(range(5)),
        outer_seeds=list(range(20)),
        outer_test_bags=list(range(1, 10)),

        output_dir="nested_production_PFOS",

        n_jobs=4,
        resume=True,
    )


if __name__ == "__main__":
    main()
