from sklearn.preprocessing import StandardScaler
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class PFASDataset(Dataset):

    def __init__(
    self,
    data_dir="data",
    use_bags=None,
    scaler=None,
    label_scaler=None
    ):

        # データフォルダの場所を保存
        self.data_dir = data_dir

        # dataフォルダ内のCSVファイルをすべて取得
        self.bag_files = sorted(
            glob.glob(os.path.join(data_dir, "*.csv"))
        )

        # PFAS.csvはラベルなので除外
        self.bag_files = [
            f for f in self.bag_files
            if "PFAS.csv" not in f
        ]

        # PFAS.csvを読み込む
        self.label_df = pd.read_csv(
        os.path.join(data_dir, "PFAS.csv")
        )

        # 空行を削除
        self.label_df = self.label_df.dropna(subset=["bag"])

        # bag番号を整数へ
        self.label_df["bag"] = self.label_df["bag"].astype(int)

        # 使用するBagだけ残す
        if use_bags is not None:
            self.label_df = self.label_df[
                self.label_df["bag"].isin(use_bags)
            ].reset_index(drop=True)
  
        # bag番号とCSVファイル名の対応
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

        # 全Bagの特徴量をまとめるリスト

        # 使用する特徴量
        feature_cols = [
            "JINKO",
            "SETAI",
            "area_m2",
            "a_住居系",
            "a_工業系",
            "a_商業系",
            "a_不明",
            "b_森林",
            "b_農地",
            "b_建物用地",
            "b_交通",
            "b_荒地",
            "b_その他人工利用地"
        ]
        # 全CSVを読み込む
        all_features = []

        for _, row in self.label_df.iterrows():

            bag = row["bag"]

            file_name = self.bag_to_file[bag]
            csv_path = os.path.join(self.data_dir, file_name)

            df = pd.read_csv(csv_path)

            # 13特徴量
            x = df[feature_cols].values

            # 追加する2特徴量
            rainfall = row["年降水量"]
            rain_days = row["降水日数"]

            rain = np.full((len(df), 1), rainfall)
            days = np.full((len(df), 1), rain_days)

            # 15特徴量へ
            x = np.hstack([x, rain, days])

            all_features.append(x)


        # 全町丁を結合
        all_features = np.vstack(all_features)

        # 特徴量のStandardScaler
        if scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(all_features)
        else:
            self.scaler = scaler

        # ラベルのStandardScaler
        labels = self.label_df[
            [
                "PFOS_ng_L",
                "PFOA_ng_L",
                "total_ng_L"
            ]
        ].values

        if label_scaler is None:
            self.label_scaler = StandardScaler()
            self.label_scaler.fit(labels)
        else:
            self.label_scaler = label_scaler



    def __len__(self):

        # 学習サンプル数を返す
        return len(self.label_df)

    def __getitem__(self, idx):

        #降水量を取得
        row = self.label_df.iloc[idx]

        rain = row["年降水量"]
        rain_days = row["降水日数"]


        # bag番号
        bag = row["bag"]

        # PFAS濃度（目的変数）
        label = np.array([
            row["PFOS_ng_L"],
            row["PFOA_ng_L"],
            row["total_ng_L"]
        ]).reshape(1, -1)

        # ラベルを標準化
        label = self.label_scaler.transform(label)[0]

        # bag番号に対応するCSVを取得
        file_name = self.bag_to_file[bag]

        # CSVのパス
        csv_path = os.path.join(self.data_dir, file_name)


        # 流域データ読込
        df = pd.read_csv(csv_path)

        # 使用する特徴量
        feature_cols = [
            "JINKO",
            "SETAI",
            "area_m2",
            "a_住居系",
            "a_工業系",
            "a_商業系",
            "a_不明",
            "b_森林",
            "b_農地",
            "b_建物用地",
            "b_交通",
            "b_荒地",
            "b_その他人工利用地"
        ]

        # 特徴量を取得
        x = df[feature_cols].values

        #降水量
        rain = np.full((len(df), 1), rain)
        rain_days = np.full((len(df), 1), rain_days)
        x = np.hstack([x, rain, rain_days])

        # 標準化
        x = self.scaler.transform(x)

        # Tensorへ変換
        x = torch.tensor(x, dtype=torch.float32)

        # ラベルもTensorへ変換
        y = torch.tensor(label, dtype=torch.float32)

        return x, y