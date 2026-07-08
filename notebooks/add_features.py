import numpy as np
import polars as pl
from polars import selectors as cs

def _add_over_all_score(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'OverallScore'
    return df.with_columns(
        (pl.col('OverallQual') + pl.col('OverallCond'))
        .alias(new_feat_name)
    )
    
def _add_bath_score(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'BathScore' 
    cols = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
    weights = np.array([2. , 1.2, 1.0, 0.5])
    mat = df.select(cols)
    score = mat @ weights
    return df.with_columns(pl.Series(new_feat_name, score))

def _add_total_flr_sf(df:pl.DataFrame)->pl.DataFrame:
    new_feat_name = 'TotalFlrSF'
    cols = ['1stFlrSF', '2ndFlrSF', 'TotalBsmtSF']
    return df.with_columns(pl.sum_horizontal(cols).alias(new_feat_name))

def _add_is_overall_ge9():
    # OverallQual 列が9以上なら1, それ以外0のラベル列
    new_feat_name = 'is_overall_ge9'
    pass

def add_modified_features(df:pl.DataFrame)->pl.DataFrame:
    functions = [
        _add_over_all_score,
        _add_bath_score,
        _add_total_flr_sf
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