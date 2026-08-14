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

drop_feats = [
    '2ndFlrSF',
    '1stFlrSF',
    'GrLivArea',
    'Exterior1st',
    'Exterior2nd',
    'TotalBsmtSF',
    'GarageCars',
    'TotalFlrSF',
    'YearBuilt',
    'OverallQual',
    'BsmtFullBath',
    'Utilities',
    'Street',
    'Condition1',
    'Condition2',
    "Fireplaces",
    "LandSlope",
]

eq_values_labeled = {
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
}

one_hot_encoded = [
    "RoofStyle",
    "Electrical",
    "GarageFinish",
]

target_encoded = [
    "MSSubClass",
    "Neighborhood",
    "HouseStyle",
    "MSZoning",
    "TargetExterior1_2",
    "BldgType",
    "LotConfig",
    "MasVnrType",
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
]

standardized = [
    "OverallCond",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageYrBlt",
    "GarageArea",
    "MoSold",
    "YrSold",
    "OverallScore",
    "BathScore",
    "IsOverAllGE9",
    # "SoldAfterRehman",
    # "SoldMay2June",
    "AreaPerRooms",
    "GarageCarRatio",
    "RemodAge",
    "BuildingAge",
    "TotalOutdoorSF",
    "BsmtFnRatio",
    "IsRemodeled",
    "LuxuryCount",
    "FireplaceScore",
    "Condition",
    "KitchenScore",
    "KitchenAbvGr",
    "MissingNormalyUtilsCount",
    "BsmtFinRatio",
    "BsmtUnfRatio",
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

    reg_ct = ColumnTransformer(
        [
            ("eq_val_label", eq_values_label_pipeline, list(eq_values_labeled.keys())),
            ("ordinal_encode", ordinal_encode_pipeline, list(ordinal_encoded)),
            ("one_hot_encode", one_hot_encode_pipeline, one_hot_encoded),
            ("target_encode", target_encode_pipeline, target_encoded),
            ("log_standardize", log_standardize_pipeline, log_standardized),
            ("standardize", standardize_pipeline, standardized),
            ("drop", "drop", drop_feats),
        ],
        remainder="drop",
    )

    tree_ct = ColumnTransformer(
        [
            ("eq_val_label", eq_values_label_pipeline, list(eq_values_labeled.keys())),
            ("ordinal_encode", ordinal_encode_pipeline, list(ordinal_encoded)),
            ("target_encode", target_encode_pipeline, target_encoded),
            ("drop", "drop", drop_feats),
        ],
        remainder="drop",
    )

    return [tree_ct, reg_ct]
