from typing import Iterable, List, Union

import numpy as np
import pandas as pd

default_features = ["VS_PS[%]", "T_ww[degC]", "HRT_AD[day]", "SRT_AS[day]", "m_PS[%]"]


def make_new_features(
    X: pd.DataFrame,
    X_features: List[str] = default_features,
    powers: Iterable[Union[int, float]] = [-2, -1, 2],
    divide: bool = True,
    mult: bool = True,
) -> pd.DataFrame:
    """
    Create new features

    Args:
        X: DataFrame on which new features are to be computed
        X_features: list of initial feature names which should be used as basis

        powers: List of exposants. For each feature F in X_feature and power p in powers,
            F ** p is created as a new feature (NOTE: The new feature will be removed if it
            contains any nan values)
        divide: boolean. If True, for each couple of different features F1, F2 in X_feature,
            F1 / F2 and F2 / F1 are added as a new feature (NOTE: The new feature will be removed
            if it contains any nan values)
        mult: boolean: If True, for each couple of different features F1, F2 in X_feature, F1 x F2 is
            added as a new feature.
    Returns:
        DataFrame containing features specified in X_features and newly created features.
    """

    X_new = X[X_features].copy()

    # Adding power
    for name in X_features:
        X_ = X_new[name]

        for i in powers:
            X_new[f"{name} ** {i}"] = X_**i

    # Adding combinations
    n = len(X_features)
    for i in range(n):
        name_i = X_features[i]
        for j in range(i + 1, n):
            name_j = X_features[j]
            if mult:
                X_new[f"{name_i} x {name_j}"] = X_new[name_i] * X_new[name_j]
            if divide:
                X_new[f"{name_i} / {name_j}"] = X_new[name_i] / X_new[name_j]
                X_new[f"{name_j} / {name_i}"] = X_new[name_j] / X_new[name_i]

    X_new.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_new = X_new.dropna(axis=1)
    return X_new.copy()  # Return copy to reorder underlying array
