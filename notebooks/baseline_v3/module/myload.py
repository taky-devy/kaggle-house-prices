"""
キャスト、欠損埋め、誤り修正、ロード(オーケストレーション)を定義する
"""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd


def _mycast(df: pd.DataFrame):
    df["MSSubClass"] = df["MSSubClass"].apply(str)
    df["YrSold"] = df["YrSold"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)
    return df


def _impute(df: pd.DataFrame):
    # ==========================
    #            None
    # ==========================
    none = [
        "Alley",
        "PoolQC",
        "MiscFeature",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "Fence",
        "FireplaceQu",
        "MasVnrType",
    ]
    for col in none:
        df[col] = df[col].fillna("None")

    # ==========================
    #            zero
    # ==========================
    zero = [
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
    ]
    for col in zero:
        df[col] = df[col].fillna(0)

    # ==========================
    #            mode
    # ==========================
    mode = [
        "Functional",
        "Electrical",
        "KitchenQual",
        "Exterior1st",
        "Exterior2nd",
        "SaleType",
        "Utilities",
    ]
    for col in mode:
        df[col] = df[col].fillna(df[col].mode()[0])

    # ==========================
    #         specific
    # ==========================
    # 同じ住宅タイプは同じ地域に集中するはず
    df["MSZoning"] = df.groupby("MSSubClass")["MSZoning"].transform(
        lambda x: x.fillna(x.mode()[0])
    )

    # 同じ区域なら道路接面長は近いはず
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # 欠損値の残存チェック
    nulls = df.isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls) > 1:  # test側の SalePriceが残るため2以上でraise
        raise ValueError(f"Impute失敗。nullを含む列: {nulls.to_dict()}")

    return df


def _clean(df: pd.DataFrame):
    # typo
    df["Exterior2nd"] = df["Exterior2nd"].replace(
        {"Brk Cmn": "BrkComm", "CmentBd": "CemntBd"}
    )

    # ありえない年代(2200年)の上書き
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)

    # clipping input : YearRemodAdd
    # 下限が1950年でクリップ入力されているためGarageYrBltの伝播を推測値とする
    is_clipped = (df["YearBuilt"] < 1950) & (df["YearRemodAdd"] == 1950)
    garage_is_zero = df["GarageYrBlt"] == 0
    conditions = [
        is_clipped & garage_is_zero,
        is_clipped & ~garage_is_zero,
    ]
    df["YearRemodAdd"] = np.select(
        conditions, [df["YearBuilt"], df["GarageYrBlt"]], default=df["YearRemodAdd"]
    )

    return df


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    # read
    dir = Path("../../data/")
    train = pd.read_csv(dir / "train.csv", index_col="Id")
    test = pd.read_csv(dir / "test.csv", index_col="Id")
    train = train.drop(
        [524, 1299]
    )  #  GrLivArea vs SalePrice 相関の有名な外れ値サンプル

    # loading pipeline (orchestration)
    df = pd.concat([train, test])
    df = _mycast(df)
    df = _impute(df)
    df = _clean(df)

    # Reform splits
    train = df.loc[train.index, :]
    test = df.loc[test.index, :]
    test = test.drop(
        columns="SalePrice"
    )  # concat->split の結果、test側にもSalePriceが増えているので消す
    train, test = cast(tuple[pd.DataFrame, pd.DataFrame], [train, test])

    # 目的変数のBin番号列を付与
    q = 5
    train["price_bin"] = pd.qcut(train["SalePrice"], q=q, labels=range(q))

    return train, test
