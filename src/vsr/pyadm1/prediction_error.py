"""Error computation for ADM1 module

Main functions:
    - adm1_err: compute error from predictions and observations
    - score_param: compute error from parameter, observations and further digester information
    - score_free_param: equivalent of score_param for FreeDigesterParameter space

score_param and score_free_param are used during calibration and UQ procedure.
"""

from typing import Union, overload

import numpy as np

from vsr.pyadm1.basic_classes import ADM1States


class ADM1Failure(Warning):
    """Warning class when ADM1 computations failed"""


@overload
def soft_plus(x: np.ndarray, max_val: float = 3.0, elbow: float = 1.0) -> np.ndarray:
    ...


@overload
def soft_plus(x: float, max_val: float = 3.0, elbow: float = 1.0) -> float:
    ...


def soft_plus(
    x: Union[np.ndarray, float], max_val: float = 3.0, elbow: float = 1.0
) -> Union[np.ndarray, float]:
    """
    Variation on the softplus function used for capping.

    Smoothed version of x -> np.max(max_val, np.abs(x)).
    args:
        x, a float to be smoothly capped (if np.ndarray, applied element wise)
        max_val, the maximum value returned as x -> infty
        elbow, a control on the smoothness

    """
    prop = (1 + np.exp(-elbow * max_val)) / elbow
    C = np.log(1 + np.exp(elbow * max_val))
    return prop * (C - np.log(1 + np.exp(elbow * (max_val - np.abs(x)))))


error_pred_names = [
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_IN",
    "q_gas",
    "q_ch4",
    "p_ch4",
    "p_co2",
]


def adm1_err(
    pred: ADM1States,
    obs: ADM1States,
    eps: float = 10 ** (-8),
    max_score: float = 3.0,
    elbow: float = 2.0,
) -> float:
    """
    Compute the error as pseudo Root mean square of log residuals

    Args:
        pred prediction from ADM1
        obs: digester observation
        eps: soft threshold for small values (i.e. if both pred and obs are << eps, the error
            contribution is close to 0)
        max_score: soft threshold for large error contribution
        elbow: smoothness of thresholding for large error contribution

    Output:
        prediction error as
            sqrt( (sum_i omega_i sum_t log(pred_{i,t}/obs_{i,t}) **2) / sum_i t_i omega_i )
        with t_i the number of non nan data for prediction type i, omega_i a renormalisation factor
        (by default 1). nan are ignored. The sum is performed only on pred_names
    """
    res = np.log(
        (pred.df[error_pred_names] + eps).to_numpy()
        / (obs.df[error_pred_names] + eps).to_numpy()
    )
    corr_res = soft_plus(res, max_val=max_score, elbow=elbow)  # type: ignore

    return np.sqrt(np.nanmean(corr_res**2))
