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
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

map_encoded = {
    "Condition1": {
        "Artery": -1,
        "Feedr": -1,
        "Norm": 0,
        "RRNn": -1,
        "RRAn": -1,
        "PosN": 1,
        "PosA": 1,
        "RRNe": -1,
        "RRAe": -1,
    },
    "Condition2": {
        "Artery": -1,
        "Feedr": -1,
        "Norm": 0,
        "RRNn": -1,
        "RRAn": -1,
        "PosN": 1,
        "PosA": 1,
        "RRNe": -1,
        "RRAe": -1,
    },
}

eq_values_labeled = {
    "Street": ["Pave"],
    "CentralAir": ["Y"],
    "Heating": ["GasA"],
    "PavedDrive": ["Y"],
    "RoofMatl": ["CompShg"],
    "Condition2": ["PosN"],
    "SaleCondition": ["Normal"],
    "LandContour": ["Lvl"],
    "SaleType": ["New"],
    "Foundation": ["BrkTil", "CBlock", "PConc"],
    "Utilities": ["AllPub"],
    "GarageType": ["Attchd", "BuiltIn"],
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
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "BsmtCond": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["None", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["None", "No", "Mn", "Av", "Gd"],
}

one_hot_encoded = [
    "LotConfig",
    "LandSlope",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "HeatingQC",
    "Electrical",
    "MasVnrType",
    "BsmtFinType1",
    "BsmtFinType2",
    "GarageFinish",
]

target_encoded = [
    "MSSubClass",
    "Neighborhood",
    "MSZoning",
    "TargetExterior1_2",
]

count_encoded = []

log_standardized = [
    "LotFrontage",
    "LotArea",
    "BsmtUnfSF",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "MasVnrArea",
    "EnclosedPorch",
    "HalfBath",
    "OpenPorchSF",
    "WoodDeckSF",
    "BsmtAboveRatio",
    "LivLotRatio",
    "LivArea_x_Qual",
    "FireplaceScore",
]

standardized = [
    "OverallCond",
    "FullBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "GarageYrBlt",
    "GarageArea",
    "MoSold",
    "YrSold",
    "OverallScore",
    "BathScore",
    "IsOverAllGE9",
    "SoldAfterRehman",
    "SoldMay2June",
    "AreaPerRooms",
    "GarageCarRatio",
    "RemodAge",
    "BuildingAge",
    "TotalOutdoorSF",
    "BsmtFnRatio",
    "IsRemodeled",
    "LuxuryCount",
]


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
            ("scale", StandardScaler()),
        ]
    )


def build_preprocessor():
    log_standardize = Pipeline(
        [
            (
                "log",
                FunctionTransformer(
                    np.log1p,
                    inverse_func=np.expm1,
                    check_inverse=False,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    map_label_pipeline = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(
                    categories=[
                        sorted(mapping, key=mapping.get)
                        for mapping in map_encoded.values()
                    ]
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    def _eq_values_transform(X, mapping=eq_values_labeled):
        X = np.asarray(X)
        out = np.empty(X.shape, dtype=int)
        for i, values in enumerate(mapping.values()):
            out[:, i] = np.isin(X[:, i], values).astype(int)
        return out

    eq_values_label_pipeline = Pipeline(
        [
            (
                "eq",
                FunctionTransformer(
                    _eq_values_transform, feature_names_out="one-to-one"
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    ordinal_encode_pipeline = Pipeline(
        [
            (
                "ordinal",
                OrdinalEncoder(categories=list(ordinal_encoded.values())),
            )
        ]
    )

    one_hot_encode_pipeline = Pipeline(
        [("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01))]
    )

    target_encode_pipeline = Pipeline(
        [
            ("target", TargetEncoder()),
            ("scale", StandardScaler()),
        ]
    )

    log_standardize_pipeline = Pipeline(
        [
            ("log_standard", log_standardize),
        ]
    )

    standardize_pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
        ]
    )

    ct_for_reg = ColumnTransformer(
        [
            ("map_label", map_label_pipeline, list(map_encoded.keys())),
            ("eq_val_label", eq_values_label_pipeline, list(eq_values_labeled.keys())),
            ("ordinal_encode", ordinal_encode_pipeline, list(ordinal_encoded)),
            ("one_hot_encode", one_hot_encode_pipeline, one_hot_encoded),
            ("target_encode", target_encode_pipeline, target_encoded),
            ("log_standardize", log_standardize_pipeline, log_standardized),
            ("standardize", standardize_pipeline, standardized),
        ],
        remainder="drop",
    )

    ct_for_tree = ColumnTransformer(
        [
            ("map_label", map_label_pipeline, list(map_encoded.keys())),
            ("eq_val_label", eq_values_label_pipeline, list(eq_values_labeled.keys())),
            ("ordinal_encode", ordinal_encode_pipeline, list(ordinal_encoded)),
            ("target_encode", target_encode_pipeline, target_encoded),
        ],
        remainder="drop",
    )

    return [ct_for_tree, ct_for_reg]
