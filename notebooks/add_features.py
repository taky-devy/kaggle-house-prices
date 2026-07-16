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

def _building_age_at_sale(df: pl.DataFrame) -> pl.DataFrame:
    # YrSold - Max(YearBuilt, YearRemodAdd)  ※ データ不備による負値を0にクリップ
    new_feat_name = 'BuildingAgeAtSale'
    return df.with_columns(
        (pl.col('YrSold').fill_null(0) - pl.max_horizontal(
            pl.col('YearBuilt').fill_null(0), pl.col('YearRemodAdd').fill_null(0)
        ))
        .clip(lower_bound=0) # 売却年の方が先で -1 (log化で-inf) になるサンプルがあるのでclipする
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
        _building_age_at_sale,
        _bsmt_above_ratio,
        _liv_lot_ratio,
        _sold_may2june,
        _sold_after_rehman,
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