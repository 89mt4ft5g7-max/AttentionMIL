import os
import pandas as pd
import torch

from dataset import PFASDataset
from model import AttentionMIL

import os

os.makedirs("attention", exist_ok=True)

# データセット
dataset = PFASDataset()

# モデル
model = AttentionMIL(input_dim=15, hidden_dim=64)

# 学習済みモデルを読み込み
model.load_state_dict(torch.load("model.pth"))
model.eval()

for i in range(len(dataset)):

    x, y = dataset[i]

    _, attention = model(x, return_attention=True)

    bag = dataset.label_df.iloc[i]["bag"]

    file_name = dataset.bag_to_file[bag]

    date = pd.to_datetime(dataset.label_df.iloc[i]["date"]).strftime("%Y-%m-%d")

    csv_path = os.path.join(dataset.data_dir, file_name)

    df = pd.read_csv(csv_path)

    df["attention"] = attention.squeeze().detach().numpy()

    #AREAの桁数固定
    df["AREA"] = df["AREA"].round(3)

    # attentionが大きい順に並べ替え
    df = df.sort_values(
        by="attention",
        ascending=False
    ).reset_index(drop=True)

    date = pd.to_datetime(
        dataset.label_df.iloc[i]["date"]
    ).strftime("%Y%m%d")

    base_name = os.path.splitext(file_name)[0]

    save_name = os.path.join(
        "attention",
        f"{base_name}_{date}_attention.csv"
    )

    df.to_csv(
        save_name,
        index=False,
        encoding="utf-8-sig"
    )