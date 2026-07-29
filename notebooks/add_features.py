import math
import numpy as np
import polars as pl
from polars import selectors as cs

def _over_all_score(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'OverallScore'
    return df.with_columns(
        (pl.col('OverallQual').fill_null(0) + pl.col('OverallCond').fill_null(0))
        .alias(new_feat_name)
    )
    
def _bath_score(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'BathScore' 
    cols = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
    weights = np.array([2. , 1.2, 1.0, 0.5])
    mat = df.select([pl.col(c).fill_null(0) for c in cols])
    score = mat @ weights
    return df.with_columns(pl.Series(new_feat_name, score))

def _total_flr_sf(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'TotalFlrSF'
    cols = ['GrLivArea', 'TotalBsmtSF']
    exprs = [pl.col(c).fill_null(0) for c in cols]
    return df.with_columns(pl.sum_horizontal(exprs).alias(new_feat_name))

def _is_overall_ge9(df: pl.DataFrame) -> pl.DataFrame:
    # OverallQual 列が9以上なら1, それ以外0のラベル列
    new_feat_name = 'IsOverAllGE9'
    return df.with_columns(
        (pl.col('OverallQual').fill_null(0) >= 9).cast(pl.Int8).alias(new_feat_name)
    )

def _remod_age(df: pl.DataFrame) -> pl.DataFrame:
    # YrSold - YearRemodAdd
    new_feat_name = 'RemodAge'
    return df.with_columns(
        (pl.col('YrSold').fill_null(strategy='mean') - pl.col('YearRemodAdd').fill_null(strategy='mean'))
        .clip(lower_bound=0)
        .alias(new_feat_name)
    )

def _building_age(df: pl.DataFrame) -> pl.DataFrame:
    # YrSold - YearBuilt
    new_feat_name = 'BuildingAge'
    return df.with_columns(
        (pl.col('YrSold').fill_null(strategy='mean') - pl.col('YearBuilt').fill_null(strategy='mean'))
        .clip(lower_bound=0)
        .alias(new_feat_name)
    )

def _bsmt_above_ratio(df: pl.DataFrame) -> pl.DataFrame:
    # TotalBsmtSF / GrLivArea
    new_feat_name = 'BsmtAboveRatio'
    return df.with_columns(
        (pl.col('TotalBsmtSF').fill_null(0) / pl.col('GrLivArea').fill_null(0))
        .fill_nan(0)
        .replace([np.inf, -np.inf], 0)
        .alias(new_feat_name)
    )

def _liv_lot_ratio(df: pl.DataFrame) -> pl.DataFrame:
    # GrLivArea / LotArea
    new_feat_name = 'LivLotRatio'
    return df.with_columns(
        (pl.col('GrLivArea').fill_null(0) / pl.col('LotArea').fill_null(0))
        .fill_nan(0)
        .replace([np.inf, -np.inf], 0)
        .alias(new_feat_name)
    )

def _sold_may2june(df: pl.DataFrame) -> pl.DataFrame:
    # MoSold列が = 5,6 なら1, それ以外 0
    new_feat_name = 'SoldMay2June'
    return df.with_columns(
        pl.col('MoSold').fill_null(0).is_in([5, 6]).cast(pl.Int8).alias(new_feat_name)
    )

def _sold_after_rehman(df: pl.DataFrame) -> pl.DataFrame:
    # リーマンショック直後(2008年10月～2009年10月)に売れたか
    # 開始日は9/15だが売却日のデータがないため月単位で2008年10月を閾値とする
    new_feat_name = 'SoldAfterRehman'
    yr_sold = pl.col('YrSold').fill_null(0)
    mo_sold = pl.col('MoSold').fill_null(0)
    months_since_epoch = yr_sold * 12 + mo_sold
    start = 2008 * 12 + 10
    end = 2009 * 12 + 10
    return df.with_columns(
        ((months_since_epoch >= start) & (months_since_epoch <= end))
        .cast(pl.Int8).alias(new_feat_name)
    )

def _are_per_rooms(df: pl.DataFrame) -> pl.DataFrame:
    # GrLivArea / TotRmsAbvGrd
    new_feat_name = 'AreaPerRooms'
    return df.with_columns(
        (pl.col('GrLivArea').fill_null(0) / pl.col('TotRmsAbvGrd').fill_null(0))
        .fill_nan(0)
        .replace([np.inf, -np.inf], 0)
        .alias(new_feat_name)
    )

def _has_garege(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = 'HasGarage'
    return df.with_columns(
        pl.when(pl.col('GarageQual') == 'NA')
        .then(0)
        .otherwise(1)
        .alias(new_feat_name)
    )

def _target_exterior1_2(df: pl.DataFrame) -> pl.DataFrame:
    # description
    new_feat_name = 'TargetExterior1_2'
    return df.with_columns(
        (pl.col('Exterior1st') + '_' + pl.col('Exterior2nd'))
        .alias(new_feat_name)
    )

def _livarea_x_qual(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = 'LivArea_x_Qual'
    return df.with_columns(
        (pl.col('OverallQual') * pl.col('GrLivArea')).alias(new_feat_name)
    )
    
def _garage_car_ratio(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = 'GarageCarRatio'
    return df.with_columns(
        (pl.col('GarageArea') / pl.col('GarageCars')).alias(new_feat_name)
    )

def _fireplace_score(df: pl.DataFrame) -> pl.DataFrame:
    new_feat_name = 'FireplaceScore'
    return df.with_columns((
        pl.col('Fireplaces').fill_null(0).sqrt() * \
        pl.when(pl.col('FireplaceQu') == 'NA')
        .then(0)
        .when(pl.col('FireplaceQu') == 'TA')
        .then(1)
        .when(pl.col('FireplaceQu') == 'Gd')
        .then(2)
        .when(pl.col('FireplaceQu') == 'Gd')
        .then(3)
        .pow(2)
        ).alias(new_feat_name)
    )

# def _hoge(df: pl.DataFrame) -> pl.DataFrame:
#     # description
#     new_feat_name = 'hoge'
#     return df.with_columns(
#     )

def add_modified_features(df:pl.DataFrame)->pl.DataFrame:
    functions = [
        _over_all_score,
        _bath_score,
        _total_flr_sf,
        _is_overall_ge9,
        _remod_age,
        _building_age,
        _bsmt_above_ratio,
        _liv_lot_ratio,
        _sold_may2june,
        _sold_after_rehman,
        _are_per_rooms,
        _has_garege,
        _target_exterior1_2,
        _livarea_x_qual,
        _garage_car_ratio,
        _fireplace_score
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
    return df