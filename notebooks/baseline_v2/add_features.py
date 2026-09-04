# TODO BsmtQual, BsmtCond, BsmtExposure の2乗和平方根 * 広さ を足す

import numpy as np
import pandas as pd
import polars as pl


def _over_all_score(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "OverallScore"
    return df.with_columns(
        (pl.col("OverallQual") + pl.col("OverallCond")).alias(new_feat_name)
    )


def _bath_score(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "BathScore"
    cols = ["BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath"]
    weights = np.array([2.0, 1.2, 1.0, 0.5])
    mat = df.select([pl.col(c) for c in cols])
    score = mat @ weights
    return df.with_columns(pl.Series(new_feat_name, score))


def _total_flr_sf(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "TotalFlrSF"
    cols = ["GrLivArea", "TotalBsmtSF"]
    return df.with_columns(pl.sum_horizontal(cols).alias(new_feat_name))


def _is_overall_ge9(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "IsOverAllGE9"
    return df.with_columns(
        (pl.col("OverallQual") >= 9).cast(pl.Int8).alias(new_feat_name)
    )


def _remod_age(df: pl.DataFrame) -> pl.DataFrame:
    # YrSold - YearRemodAdd
    new_feat_name = "RemodAge"
    return df.with_columns(
        (
            pl.col("YrSold").cast(pl.Int32).fill_null(strategy="mean")
            - pl.col("YearRemodAdd").fill_null(strategy="mean")
        )
        .clip(lower_bound=0)  # 販売より後に施工するパターンに対応
        .alias(new_feat_name)
    )


def _building_age(df: pl.DataFrame) -> pl.DataFrame:
    # YrSold - YearBuilt
    new_feat_name = "BuildingAge"
    return df.with_columns(
        (
            pl.col("YrSold").cast(pl.Int32).fill_null(strategy="mean")
            - pl.col("YearBuilt").fill_null(strategy="mean")
        )
        .clip(lower_bound=0)  # 販売より後に完成するパターンに対応
        .alias(new_feat_name)
    )


def _bsmt_unf_ratio(df: pl.DataFrame) -> pl.DataFrame:
    # BsmtUnfSF / TotalBsmtSF
    new_feat_name = "BsmtUnfRatio"
    return df.with_columns(
        (pl.col("BsmtUnfSF").fill_null(0) / pl.col("TotalBsmtSF").fill_null(0))
        .fill_nan(0)
        .replace([np.inf, -np.inf], 0)
        .alias(new_feat_name)
    )


def _no_bsmt(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "NoBsmt"
    return df.with_columns(
        pl.when(pl.col("TotalBsmtSF") == 0).then(1).otherwise(0).alias(new_feat_name)
    )


def _bsmt_above_ratio(df: pl.DataFrame) -> pl.DataFrame:
    # TotalBsmtSF / GrLivArea
    new_feat_name = "BsmtAboveRatio"
    return df.with_columns(
        (pl.col("TotalBsmtSF").fill_null(0) / pl.col("GrLivArea").fill_null(0))
        .fill_nan(0)
        .replace([np.inf, -np.inf], 0)
        .alias(new_feat_name)
    )


def _is_culdsac(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "IsCuldsac"
    return df.with_columns(
        pl.when(pl.col("LotConfig") == "CulDSac")
        .then(1)
        .otherwise(0)
        .alias(new_feat_name)
    )


def _liv_lot_ratio(df: pl.DataFrame) -> pl.DataFrame:
    # GrLivArea / LotArea
    new_feat_name = "LivLotRatio"
    return df.with_columns(
        (pl.col("GrLivArea").fill_null(0) / pl.col("LotArea").fill_null(0))
        .fill_nan(0)
        .replace([np.inf, -np.inf], 0)
        .alias(new_feat_name)
    )


def _sold_may2june(df: pl.DataFrame) -> pl.DataFrame:
    # MoSold列が = 5,6 （引っ越しシーズン）なら1, それ以外 0
    new_feat_name = "SoldMay2June"
    return df.with_columns(
        pl.col("MoSold").fill_null(0).is_in([5, 6]).cast(pl.Int8).alias(new_feat_name)
    )


def _sold_after_rehman(df: pl.DataFrame) -> pl.DataFrame:
    # リーマンショック直後(2008年10月～2008年12月)に売れたか
    # 開始日は9/15だが売却日のデータがないため月単位で2008年10月を閾値とする
    # 下のAmes市の住宅価格指数では実はそんなに影響なかった（ほぼ横ばい）ので意味なさそう
    # https://fred.stlouisfed.org/series/ATNHPIUS11180Q
    new_feat_name = "SoldAfterRehman"
    yr_sold = pl.col("YrSold").fill_null(0)
    mo_sold = pl.col("MoSold").fill_null(0)
    months_since_epoch = yr_sold * 12 + mo_sold
    start = 2008 * 12 + 10
    end = 2008 * 12 + 10
    return df.with_columns(
        ((months_since_epoch >= start) & (months_since_epoch <= end))
        .cast(pl.Int8)
        .alias(new_feat_name)
    )


def _area_per_rooms(df: pl.DataFrame) -> pl.DataFrame:
    # GrLivArea / TotRmsAbvGrd
    new_feat_name = "AreaPerRooms"
    return df.with_columns(
        (pl.col("GrLivArea").fill_null(0) / pl.col("TotRmsAbvGrd").fill_null(0))
        .fill_nan(0)
        .replace([np.inf, -np.inf], 0)
        .alias(new_feat_name)
    )


def _no_garege(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "NoGarage"
    return df.with_columns(
        pl.when(pl.col("GarageQual") == "None")
        .then(1)
        .otherwise(0)
        .alias(new_feat_name)
    )


def _target_exterior1_2(df: pl.DataFrame) -> pl.DataFrame:
    # description
    new_feat_name = "TargetExterior1_2"
    return df.with_columns(
        (pl.col("Exterior1st") + "_" + pl.col("Exterior2nd")).alias(new_feat_name)
    )


def _livarea_x_qual(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "LivArea_x_Qual"
    return df.with_columns(
        (pl.col("OverallQual") * pl.col("GrLivArea")).alias(new_feat_name)
    )


def _garage_car_ratio(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "GarageCarRatio"
    return df.with_columns(
        (pl.col("GarageArea") / pl.col("GarageCars")).fill_nan(0).alias(new_feat_name)
    )


def _fireplace_score(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "FireplaceScore"
    qual_map = {"None": 1, "Po": 2, "Fa": 3, "TA": 4, "Gd": 5, "Ex": 6}
    return df.with_columns(
        (
            pl.col("Fireplaces").sqrt()
            * pl.col("FireplaceQu").replace(qual_map).cast(pl.Int8)
        )
        .fill_null(0)
        .alias(new_feat_name)
    )


def _total_outdoor_sf(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "TotalOutdoorSF"
    return df.with_columns(
        pl.sum_horizontal(
            "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch"
        ).alias(new_feat_name)
    )


def _bsmt_score(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "BsmtScore"
    qual_map = {"None": 1, "Po": 2, "Fa": 3, "TA": 4, "Gd": 5, "Ex": 6}
    cond_map = {"None": 1, "Po": 2, "Fa": 3, "TA": 4, "Gd": 5, "Ex": 6}
    expo_map = {"None": 1, "No": 2, "Mn": 3, "Av": 4, "Gd": 5}
    return df.with_columns(
        (
            pl.col("TotalBsmtSF")
            * pl.mean_horizontal(
                pl.col("BsmtQual").replace(qual_map).cast(pl.Int8),
                pl.col("BsmtCond").replace(cond_map).cast(pl.Int8),
                pl.col("BsmtExposure").replace(expo_map).cast(pl.Int8),
            )
        ).alias(new_feat_name)
    )


def _is_remodeled(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "IsRemodeled"
    return df.with_columns(
        (
            pl.when(pl.col("YearRemodAdd") > pl.col("YearBuilt").cast(pl.Int32))
            .then(1)
            .otherwise(0)
        ).alias(new_feat_name)
    )


def _luxury_count(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "LuxuryCount"
    return df.with_columns(
        (
            pl.when(pl.col("PoolArea") > 0).then(1).otherwise(0)
            + pl.when(pl.col("FireplaceQu").is_in(["Gd", "Ex"])).then(1).otherwise(0)
            + pl.when(pl.col("MiscFeature") == "TenC").then(1).otherwise(0)
            + pl.when(pl.col("GarageCars") >= 3).then(1).otherwise(0)
            + pl.when(pl.col("KitchenQual") == "Ex").then(1).otherwise(0)
            + pl.when(pl.col("BsmtQual") == "Ex").then(1).otherwise(0)
            + pl.when(pl.col("HeatingQC") == "Ex").then(1).otherwise(0)
        ).alias(new_feat_name)
    )


def _condition(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "Condition"
    cond_map = {
        "Artery": -1,
        "Feedr": -1,
        "Norm": 0,
        "RRNn": -1,
        "RRAn": -1,
        "PosN": 1,
        "PosA": 1,
        "RRNe": -1,
        "RRAe": -1,
    }
    return df.with_columns(
        (
            pl.col("Condition1").replace(cond_map).cast(pl.Int8)
            + pl.col("Condition2").replace(cond_map).cast(pl.Int8)
        ).alias(new_feat_name)
    )


def _kitchen_score(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "KitchenScore"
    qual_map = {"None": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
    return df.with_columns(
        (
            pl.col("KitchenAbvGr").fill_nan(0)
            * pl.col("KitchenQual").replace(qual_map).cast(pl.Int8)
        ).alias(new_feat_name)
    )


def _lower_bldg_types(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "LowerBldgTypes"
    return df.with_columns(
        (
            pl.when(~pl.col("BldgType").is_in(["1Fam", "TwnhsE"])).then(1).otherwise(0)
        ).alias(new_feat_name)
    )


def _missing_normaly_utils_count(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "MissingNormalyUtilsCount"
    return df.with_columns(
        (
            pl.when(pl.col("CentralAir") == "None").then(1).otherwise(0)
            + pl.when(pl.col("GarageType") == "None").then(1).otherwise(0)
            + pl.when(pl.col("BsmtQual") == "None").then(1).otherwise(0)
            + pl.when(pl.col("PavedDrive").is_in(["N", "P"])).then(1).otherwise(0)
        )
        .fill_null(0)
        .alias(new_feat_name)
    )


def _expensive_neighborhoods(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "ExNeighborhoods"
    return df.with_columns(
        pl.when(
            pl.col("Neighborhood").is_in(
                [
                    "StoneBr",
                    "NoRidge",
                    "NridgHt",  # グループ平均トップ3
                ]
            )
        )
        .then(1)
        .otherwise(0)
        .alias(new_feat_name)
    )


# 1階に対する2階の面積比率
def _2nd_1st_flr_ratio(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = "2nd1stFlrRatio"
    return df.with_columns(
        (pl.col("2ndFlrSF") / pl.col("1stFlrSF"))
        .fill_nan(0)
        .replace([np.inf, -np.inf], 0)
        .alias(new_feat_name)
    )


# def _hoge(df: pl.DataFrame) -> pl.DataFrame:
#     new_feat_name = ''
#     return df.with_columns(
#     )


def add_modified_features(df: pl.DataFrame) -> pd.DataFrame:
    functions = [
        _over_all_score,
        _bath_score,
        _total_flr_sf,
        _is_overall_ge9,
        _remod_age,
        _building_age,
        _bsmt_above_ratio,
        _liv_lot_ratio,
        # _sold_may2june,  効いてなさげ
        # _sold_after_rehman,  当時の価格指数が横ばいなので意味なさそう
        _area_per_rooms,
        _no_garege,
        _target_exterior1_2,
        _livarea_x_qual,
        _garage_car_ratio,
        _fireplace_score,
        _total_outdoor_sf,
        _is_remodeled,
        _luxury_count,
        _condition,
        _kitchen_score,
        _lower_bldg_types,
        _missing_normaly_utils_count,
        _bsmt_unf_ratio,
        _bsmt_score,
        _no_bsmt,
        _is_culdsac,
        _2nd_1st_flr_ratio,
        _expensive_neighborhoods,
    ]

    for f in functions:
        try:
            df = f(df)
        except pl.exceptions.ColumnNotFoundError as e:
            raise pl.exceptions.ColumnNotFoundError(
                f"[{f.__name__}] 必要なカラムが見つかりません: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"[{f.__name__}] 特徴量の生成中にエラーが発生しました: {e}"
            ) from e
    return df.to_pandas()
