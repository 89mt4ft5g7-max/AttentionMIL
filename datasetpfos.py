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
        self.rainfall_cols = [
            "1日前降水量", "7日間降水量", "30日間降水量",
        ]
        self.all_feature_cols = self.feature_cols + self.rainfall_cols
        self.input_dim = len(self.all_feature_cols)

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
        missing_rainfall_cols = [
            col for col in self.rainfall_cols if col not in self.label_df.columns
        ]
        if missing_rainfall_cols:
            raise ValueError(
                f"短期降雨特徴量の列がありません: {missing_rainfall_cols}"
            )

        self.label_df[self.rainfall_cols] = self.label_df[
            self.rainfall_cols
        ].apply(pd.to_numeric, errors="raise")
        rainfall_nan_counts = self.label_df[self.rainfall_cols].isna().sum()
        if rainfall_nan_counts.any():
            raise ValueError(
                "短期降雨特徴量にNaNがあります: "
                f"{rainfall_nan_counts[rainfall_nan_counts > 0].to_dict()}"
            )

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
                x = self._build_features(df, row)
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

    def _build_features(self, df, row):
        x = df[self.feature_cols].to_numpy(dtype=np.float64)
        rainfall = np.tile(
            row[self.rainfall_cols].to_numpy(dtype=np.float64),
            (len(df), 1),
        )
        return np.hstack([x, rainfall])

    def __getitem__(self, idx):
        row = self.label_df.iloc[idx]
        bag = row["bag"]

        # ラベル処理
        label = np.array([[row["PFOS_ng_L"]]])
        label = self.label_scaler.transform(label)

        # 特徴量処理（キャッシュから取得）
        df = self.bag_data[bag]
        x = self._build_features(df, row)

        x = self.scaler.transform(x)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )
