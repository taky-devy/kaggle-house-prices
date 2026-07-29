import numpy as np
from category_encoders import CountEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    Binarizer,
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


map_label = {
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

eq_values_label = {
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

is_na_label = [
    "Alley",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

is_zero_label = [
    "LowQualFinSF",
    "PoolArea",
    "BsmtHalfBath",
    "ScreenPorch",
    "3SsnPorch",
    "MiscVal",
]

ordinal_encoded_data_error = {
    "KitchenQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Functional": ["Typ", "Min1", "Min2", "Mod", "Maj1", "Maj2", "Sev", "Sal"],
    "ExterQual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond": ["Po", "Fa", "TA", "Gd", "Ex"],
}

ordinal_encoded_no_feature = {
    "FireplaceQu": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageQual": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "BsmtCond": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtQual": ["Missing", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure": ["Missing", "No", "Mn", "Av", "Gd"],
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
]


def get_feature_config():
    return {
        "map_label": map_label,
        "eq_values_label": eq_values_label,
        "is_na_label": is_na_label,
        "is_zero_label": is_zero_label,
        "ordinal_encoded_data_error": ordinal_encoded_data_error,
        "ordinal_encoded_no_feature": ordinal_encoded_no_feature,
        "one_hot_encoded": one_hot_encoded,
        "target_encoded": target_encoded,
        "count_encoded": count_encoded,
        "log_standardized": log_standardized,
        "standardized": standardized,
    }


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
                OrdinalEncoder(categories=[sorted(mapping, key=mapping.get) for mapping in map_label.values()]),
            ),
            ("scale", StandardScaler()),
        ]
    )

    def _eq_values_transform(X, mapping=eq_values_label):
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
                    _eq_values_transform,
                    feature_names_out="one-to-one",
                ),
            ),
            ("scale", StandardScaler()),
        ]
    )

    is_na_label_pipeline = Pipeline(
        [
            ("indicator", MissingIndicator(features="all", missing_values=None)),
            ("scale", StandardScaler()),
        ]
    )

    is_zero_label_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value=0)),
            ("binarize", Binarizer(threshold=0)),
            ("scale", StandardScaler()),
        ]
    )

    ordinal_data_error_sign = np.array([-1 if col == "Functional" else 1 for col in ordinal_encoded_data_error])

    ordinal_encoded_data_error_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", missing_values=None)),
            ("ordinal", OrdinalEncoder(categories=list(ordinal_encoded_data_error.values()))),
            (
                "sign",
                FunctionTransformer(
                    lambda X: np.asarray(X) * ordinal_data_error_sign,
                    feature_names_out="one-to-one",
                ),
            ),
        ]
    )

    ordinal_encoded_no_feature_pipeline = Pipeline(
        [
            (
                "impute",
                SimpleImputer(strategy="constant", fill_value="Missing", missing_values=None),
            ),
            ("ordinal", OrdinalEncoder(categories=list(ordinal_encoded_no_feature.values()))),
        ]
    )

    one_hot_encoded_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", missing_values=None)),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]
    )

    target_encoded_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", missing_values=None)),
            ("target", TargetEncoder()),
            ("scale", StandardScaler()),
        ]
    )

    count_encoded_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent", missing_values=None)),
            ("count", CountEncoder()),
        ]
    )

    log_standardized_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log_standard", log_standardize),
        ]
    )

    standardized_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        [
            ("map_label", map_label_pipeline, list(map_label.keys())),
            ("eq_val_label", eq_values_label_pipeline, list(eq_values_label.keys())),
            ("is_na_label", is_na_label_pipeline, is_na_label),
            ("is_zero_label", is_zero_label_pipeline, is_zero_label),
            ("ordinal_encoded_data_error", ordinal_encoded_data_error_pipeline, list(ordinal_encoded_data_error)),
            ("ordinal_encoded_no_feature", ordinal_encoded_no_feature_pipeline, list(ordinal_encoded_no_feature)),
            ("one_hot_encoded", one_hot_encoded_pipeline, one_hot_encoded),
            ("target_encoded", target_encoded_pipeline, target_encoded),
            ("count_encoded", count_encoded_pipeline, count_encoded),
            ("log_standardized", log_standardized_pipeline, log_standardized),
            ("standardized", standardized_pipeline, standardized),
        ],
        remainder="drop",
    )
