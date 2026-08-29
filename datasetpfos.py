import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class PFASDataset(Dataset):

    def __init__(
        self,
        use_bags,
        data_dir="data",
        label_csv="data/PFAS.csv",
        scaler=None,
        label_scaler=None
    ):
        self.data_dir = data_dir

        # 使用する特徴量カラム
        self.feature_cols = [
            "JINKO", "SETAI", "area_m2",
            "a_住居系", "a_工業系", "a_商業系", "a_不明",
            "b_森林", "b_農地", "b_建物用地", "b_交通", "b_荒地", "b_その他人工利用地",
        ]

        # dataset.py に合わせた bag番号とCSVファイル名の対応
        self.bag_to_file = {
            1: "simotutidana.csv",
            2: "mutuai.csv",
            3: "yabune.csv",
            4: "tatikawa.csv",
            5: "kitiue.csv",
            6: "fukuda1.csv",
            7: "fukuda.csv",
            8: "kusayanagi.csv",
            9: "fujimi.csv"
        }

        # ラベル読込 ＆ フィルタリング
        self.label_df = pd.read_csv(label_csv)
        self.label_df = self.label_df.dropna(subset=["bag"])
        self.label_df["bag"] = self.label_df["bag"].astype(int)
        self.label_df = self.label_df[
            self.label_df["bag"].isin(use_bags)
        ].reset_index(drop=True)

        # CSVデータをメモリにあらかじめキャッシュ（高速化）
        self.bag_data = {}
        for bag_id in use_bags:
            file_path = os.path.join(self.data_dir, self.bag_to_file[bag_id])
            self.bag_data[bag_id] = pd.read_csv(file_path)

        # -----------------------------
        # 特徴量 Scaler
        # -----------------------------
        if scaler is None:
            X = []
            for _, row in self.label_df.iterrows():
                bag = row["bag"]
                df = self.bag_data[bag]
                x = df[self.feature_cols].values

                rain = np.full((len(df), 1), row["年降水量"])
                rain_days = np.full((len(df), 1), row["降水日数"])
                x = np.hstack([x, rain, rain_days])
                X.append(x)

            X = np.vstack(X)
            self.scaler = StandardScaler()
            self.scaler.fit(X)
        else:
            self.scaler = scaler

        # -----------------------------
        # ラベル Scaler (PFOS専用)
        # -----------------------------
        if label_scaler is None:
            self.label_scaler = StandardScaler()
            labels = self.label_df[["PFOS_ng_L"]].values
            self.label_scaler.fit(labels)
        else:
            self.label_scaler = label_scaler

    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        bag = row["bag"]

        # ラベル処理
        label = np.array([[row["PFOS_ng_L"]]])
        label = self.label_scaler.transform(label)

        # 特徴量処理（キャッシュから取得）
        df = self.bag_data[bag]
        x = df[self.feature_cols].values

        rain = np.full((len(df), 1), row["年降水量"])
        rain_days = np.full((len(df), 1), row["降水日数"])
        x = np.hstack([x, rain, rain_days])

        x = self.scaler.transform(x)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )