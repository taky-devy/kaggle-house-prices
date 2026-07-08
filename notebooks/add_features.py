import numpy as np
import polars as pl
from polars import selectors as cs

def _over_all_score(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'OverallScore'
    return df.with_columns(
        (pl.col('OverallQual') + pl.col('OverallCond'))
        .alias(new_feat_name)
    )
    
def _bath_score(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'BathScore' 
    cols = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
    weights = np.array([2. , 1.2, 1.0, 0.5])
    mat = df.select(cols)
    score = mat @ weights
    return df.with_columns(pl.Series(new_feat_name, score))

def _total_flr_sf(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'TotalFlrSF'
    cols = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
    return df.with_columns(pl.sum_horizontal(cols).alias(new_feat_name))

def _is_overall_ge9(df: pl.DataFrame) -> pl.DataFrame:
    # OverallQual 列が9以上なら1, それ以外0のラベル列
    new_feat_name = 'IsOverAllGE9'
    return df.with_columns(
        (pl.col('OverallQual') >= 9).cast(pl.Int8).alias(new_feat_name)
    )

def _building_age_at_sale(df: pl.DataFrame) -> pl.DataFrame:
    # YrSold - Max(YearBuilt, YearRemodAdd)  ※ データ不備による負値を0にクリップ
    new_feat_name = 'BuildingAgeAtSale'
    return df.with_columns(
        (pl.col('YrSold') - pl.max_horizontal('YearBuilt', 'YearRemodAdd'))
        .clip(lower_bound=0) # 売却年の方が先で -1 (log化で-inf) になるサンプルがあるのでclipする
        .alias(new_feat_name)
    )

def _bsmt_above_ratio(df: pl.DataFrame) -> pl.DataFrame:
    # TotalBsmtSF / (1stFlrSF + 2ndFlrSF)
    new_feat_name = 'BsmtAboveRatio'
    return df.with_columns(
        (pl.col('TotalBsmtSF') / (pl.col('1stFlrSF') + pl.col('2ndFlrSF')))
        .alias(new_feat_name)
    )

def _liv_lot_ratio(df: pl.DataFrame) -> pl.DataFrame:
    # GrLivArea / LotArea
    new_feat_name = 'LivLotRatio'
    return df.with_columns(
        (pl.col('GrLivArea') / pl.col('LotArea')).alias(new_feat_name)
    )

def add_modified_features(df:pl.DataFrame)->pl.DataFrame:
    functions = [
        _over_all_score,
        _bath_score,
        _total_flr_sf,
        _is_overall_ge9,
        _building_age_at_sale,
        _bsmt_above_ratio,
        _liv_lot_ratio,
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