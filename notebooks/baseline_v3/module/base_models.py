import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.pipeline import FunctionTransformer, make_pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor


def _define_base_models() -> dict:

    base_models = {}

    # liniear
    base_models["enet"] = make_pipeline(
        RobustScaler(), ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=711)
    )
    base_models["lasso"] = make_pipeline(
        RobustScaler(), Lasso(alpha=0.001, random_state=711)
    )

    # gbm
    base_models["xgb"] = XGBRegressor(
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=711,
        n_jobs=-1,
    )
    base_models["lgbm"] = LGBMRegressor(
        objective="regression",
        learning_rate=0.05,
        reg_alpha=0.0,
        reg_lambda=1.0,
        max_depth=5,
        num_leaves=31,
        colsample_bytree=0.8,
        subsample=0.8,
        min_child_samples=1,
        verbosity=-1,
    )

    return base_models


def provide_base_models() -> dict:

    models = _define_base_models()

    log_transformer = FunctionTransformer(
        np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
        feature_names_out="one-to-one",
    )

    models = {
        name: TransformedTargetRegressor(
            regressor=model,
            transformer=log_transformer,
            check_inverse=False,
        )
        for name, model in models.items()
    }

    return models
