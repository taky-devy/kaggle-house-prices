"""
# description
    前処理のパイプラインを生成する

# やること
    - 列 × 前処理系 のマッピング
    - 木モデル用のColumnTransformerの生成
    - 回帰モデル用のColumnTransformerの生成
    - TargetTransformerの生成
# やらないこと(理由)
    - impute (ipynb側に委ねるため)
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    TargetEncoder,
)

# 各特徴と前処理のマッピング定義
drop_feats = [
    # より説明的な生の特徴があるため
    "HouseStyle",  # MSSubClassが内包している
    # 追加した特徴に統合したため (多重共線性を懸念)
    "2ndFlrSF",  # GrLivArea
    "1stFlrSF",  # GrLivArea
    "GrLivArea",  # TotalFlrSF
    "TotalBsmtSF",  # TotalFlrSF
    "Exterior1st",  # Exterior1_2
    "Exterior2nd",  # Exterior1_2
    "BsmtFullBath",  # BathScore
    "Fireplaces",  # FireplaceScore
    "OverallQual",  # OverallScore, LivArea_x_Qual
    "Condition1",  # Condition
    "Condition2",  # Condition
    # より説明的な特徴を追加したため
    "YearBuilt",  # RemodAge, BuildingAge
    "MoSold",
    "YrSold",
    # 欠損多すぎのため
    "Street",
    # コンペディスカッションや参考ノートにて寄与が低いとされていたため
    "Utilities",
    # 説明コストが重そうなので保留
    # LandSlope
    #   傾斜地はマイナス要素っぽいが景観でプラスに働くことがあるらしい
    #   平坦はプラス要素っぽいが計画区域の安物は平地に多いとも考えられる
    "LandSlope",
    "RoofStyle",
]

eq_values_labeled = {
    "Electrical": ["SBrkr"],
    "Street": ["Pave"],
    "CentralAir": ["Y"],
    "Heating": ["GasA"],
    "PavedDrive": ["Y"],
    "RoofMatl": ["CompShg"],
    "SaleCondition": ["Abnormal"],
    "LandContour": ["Lvl"],
    "SaleType": ["New"],
    "Foundation": ["BrkTil", "CBlock", "PConc"],
    "Utilities": ["AllPub"],
    "GarageType": ["Attchd", "BuiltIn"],
    # "BldgType": ["1Fam","TwnhsE"],
}

ordinal_encoded = {
    "KitchenQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "Functional": [
        "None",
        "Sal",
        "Sev",
        "Maj2",
        "Maj1",
        "Mod",
        "Min2",
        "Min1",
        "Typ",
    ],
    "ExterQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "FireplaceQu": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "HeatingQC": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["None", "Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "GarageFinish": ["None", "Unf", "RFn", "Fin"],
}

one_hot_encoded = [
    "LotConfig",
]

target_encoded = [
    "MSSubClass",
    "Neighborhood",
    "MSZoning",
    "TargetExterior1_2",
    "BldgType",
    "MasVnrType",
]

log_standardized = [
    "LotFrontage",
    "LotArea",
    "BsmtUnfSF",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "MasVnrArea",
    "EnclosedPorch",
    "OpenPorchSF",
    "WoodDeckSF",
    "BsmtAboveRatio",
    "LivLotRatio",
    "LivArea_x_Qual",
    "BsmtUnfRatio",
    "TotalOutdoorSF",
]

standardized = [
    "OverallCond",
    "TotalFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageCars",
    "GarageYrBlt",
    "GarageArea",
    "OverallScore",
    "BathScore",
    "AreaPerRooms",
    "GarageCarRatio",
    "RemodAge",
    "BuildingAge",
    "IsRemodeled",
    "LuxuryCount",
    "FireplaceScore",
    "Condition",
    "KitchenScore",
    "KitchenAbvGr",
    "MissingNormalyUtilsCount",
    "HalfBath",
    "BsmtScore",
]


# TargetTransformerの定義
def build_target_transformer():
    return Pipeline(
        [
            (
                "log",
                FunctionTransformer(
                    np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                ),
            ),
            ("scale", RobustScaler()),
        ]
    )


# Column(Feature)Transformerの定義
def build_preprocessor():

    log_transformer = FunctionTransformer(
        np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
        feature_names_out="one-to-one",
    )

    def _eq_values_transform(X, mapping=eq_values_labeled):
        X = np.asarray(X)
        out = np.empty(X.shape, dtype=int)
        for i, values in enumerate(mapping.values()):
            out[:, i] = np.isin(X[:, i], values).astype(int)
        return out

    eq_values_transformer = FunctionTransformer(
        _eq_values_transform, feature_names_out="one-to-one"
    )

    ordinal_encoder = OrdinalEncoder(categories=list(ordinal_encoded.values()))

    one_hot_encoder = OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)

    reg_ct = ColumnTransformer(
        [
            (
                "eq_val_label",
                Pipeline(
                    [
                        ("label", eq_values_transformer),
                        ("scale", RobustScaler()),
                    ]
                ),
                list(eq_values_labeled.keys()),
            ),
            (
                "ordinal_encode",
                Pipeline(
                    [
                        ("encode", ordinal_encoder),
                        ("scale", RobustScaler()),
                    ]
                ),
                list(ordinal_encoded),
            ),
            (
                "one_hot_encode",
                Pipeline(
                    [
                        ("encode", one_hot_encoder),
                        ("scale", RobustScaler(with_centering=False)),
                    ]
                ),
                one_hot_encoded,
            ),
            (
                "target_encode",
                Pipeline(
                    [
                        ("encode", TargetEncoder(target_type="continuous")),
                        ("scale", RobustScaler()),
                    ]
                ),
                target_encoded,
            ),
            (
                "log",
                Pipeline(
                    [
                        ("log", log_transformer),
                        ("scale", RobustScaler()),
                    ]
                ),
                log_standardized,
            ),
            ("scale", RobustScaler(), standardized),
            ("drop", "drop", drop_feats),
        ],
        remainder="drop",
    )

    tree_ct = ColumnTransformer(
        [
            ("eq_val_label", eq_values_transformer, list(eq_values_labeled.keys())),
            ("ordinal_encode", ordinal_encoder, list(ordinal_encoded)),
            ("one_hot_encode", one_hot_encoder, one_hot_encoded),
            ("target_encode", TargetEncoder(target_type="continuous"), target_encoded),
            ("log", "passthrough", log_standardized),
            ("scale", "passthrough", standardized),
            ("drop", "drop", drop_feats),
        ],
        remainder="drop",
    )

    return [tree_ct, reg_ct]
