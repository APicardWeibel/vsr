from typing import Any

import numpy as np
import pandas as pd
from vsr.misc.sets import are_set_equal


def get_last_valid_index(df: pd.DataFrame) -> tuple[int, Any]:
    """From dataframe df with potential nan value,
    return the highest row location count such that
    df.iloc[0:count] has no Nan, as well as the index of the value
    df.index[count-1] (None if count = 0)"""
    df_nan = np.isnan(df.to_numpy())
    index_nan = np.any(df_nan, axis=1)

    # First index is not valid
    if index_nan[0]:
        return 0, None

    # All indexes are valid
    if np.all(~index_nan):
        return len(index_nan), df.index[-1]

    # At least a good index (0) and at least a bad index
    count = np.argmin(~index_nan)
    return count, df.index[count - 1]


# EQUALITY CHECKS
def frames_is_close(
    df_1: pd.DataFrame, df_2: pd.DataFrame, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """
    Checks if two dataframes are almost equal

    Args:
        df_1, df_2: the two dataframes to be compared to one another
        kwargs: rtol, atol, passed to numpy allclose function

    TODO:
        Check function. Notably, check behavior if dataframes contain non
        numerical columns, or have inadequate dtype (object)
    """

    if not are_set_equal(df_1.index, df_2.index):
        return False
    if not are_set_equal(df_1.columns, df_2.columns):
        return False

    return np.allclose(
        df_1, df_2.loc[df_1.index][df_1.columns], equal_nan=True, rtol=rtol, atol=atol
    )


def frames_coincide(
    df_1: pd.DataFrame, df_2: pd.DataFrame, rtol: float = 1e-5, atol: float = 1e-8
) -> bool:
    """Check that df_2 coincide with df_1 on df_1 index
    Args:
        df_1, df_2: Two dataframes
    Returns:
        True if df_1's index is contained in df_2 and if df_1 and df_2 coincide
        on this index, False if not
    """
    idx_min = df_1.index
    if set(idx_min).difference(df_2.index):
        # index of df_1 is not contained in df_2
        return False
    return frames_is_close(df_1, df_2.loc[idx_min], atol=atol, rtol=rtol)


def frames_equal(df_1: pd.DataFrame, df_2: pd.DataFrame):
    if not are_set_equal(df_1.index, df_2.index):
        return False
    if not are_set_equal(df_1.columns, df_2.columns):
        return False

    return ((df_1 - df_2) == 0.0).all().all()
