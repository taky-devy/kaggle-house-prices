from pathlib import Path
from typing import cast

import pandas as pd


def mycast(df: pd.DataFrame):
    df["MSSubClass"] = df["MSSubClass"].apply(str)
    df["YrSold"] = df["YrSold"].astype(str)
    df["MoSold"] = df["MoSold"].astype(str)
    return df


def clean(df: pd.DataFrame):
    # maybe typo
    df["Exterior2nd"] = df["Exterior2nd"].replace(
        {"Brk Cmn": "BrkComm", "CmentBd": "CemntBd"}
    )
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    return df


def impute(df):
    # suitable
    df["Functional"] = df["Functional"].fillna("Typ")
    df["Electrical"] = df["Electrical"].fillna("SBrkr")
    df["KitchenQual"] = df["KitchenQual"].fillna("TA")
    df["PoolQC"] = df["PoolQC"].fillna("None")
    # mode
    df["Exterior1st"] = df["Exterior1st"].fillna(df["Exterior1st"].mode()[0])
    df["Exterior2nd"] = df["Exterior2nd"].fillna(df["Exterior2nd"].mode()[0])
    df["SaleType"] = df["SaleType"].fillna(df["SaleType"].mode()[0])
    # zero
    for col in ("GarageYrBlt", "GarageArea", "GarageCars"):
        df[col] = df[col].fillna(0)
    # None
    for col in ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]:
        df[col] = df[col].fillna("None")
    for col in ("BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"):
        df[col] = df[col].fillna("None")

    # group mean
    # 同じ住宅タイプは同じ地域に集中する(都市計画的な意味で)だろうというアイデアに基づく
    df["MSZoning"] = df.groupby("MSSubClass")["MSZoning"].transform(
        lambda x: x.fillna(x.mode()[0])
    )

    # 残りの str 型は一律 None
    non_numeric_cols = df.select_dtypes(exclude="number").columns
    df[non_numeric_cols] = df[non_numeric_cols].fillna("None")

    # median
    df["LotFrontage"] = df.groupby("Neighborhood")["LotFrontage"].transform(
        lambda x: x.fillna(x.median())
    )

    # 残りの numerics 型は一律 0
    numeric_dtypes = ["int16", "int32", "int64", "float16", "float32", "float64"]
    numerics = []
    for i in df.columns:
        if df[i].dtype in numeric_dtypes:
            numerics.append(i)
    df.update(df[numerics].fillna(0))
    return df


def load_data():
    # Read
    dir = Path("../../data/")
    train = pd.read_csv(dir / "train.csv", index_col="Id")
    test = pd.read_csv(dir / "test.csv", index_col="Id")
    # filter outlier
    outlier_ids = [524, 1299]  # 確認した外れ値のId
    train = train.drop(outlier_ids)
    # Preprocessing
    df = pd.concat(
        [train, test],
    )
    df = clean(df)
    df = impute(df)
    df = mycast(df)
    # Reform splits
    train = df.loc[train.index, :]
    test = df.loc[test.index, :]
    train, test = cast(tuple[pd.DataFrame, pd.DataFrame], [train, test])
    return train, test
