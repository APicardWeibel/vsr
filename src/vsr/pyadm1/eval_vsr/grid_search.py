from typing import Iterable, Optional

import numpy as np
import pandas as pd
import scipy.stats as scs
from vsr.misc import par_eval
from vsr.pyadm1.eval_vsr.read_data import PreDig

B0_prim_s = np.linspace(200, 500, 31, endpoint=True)
B0_bio_s = np.linspace(100, 350, 26, endpoint=True)
k_dis_s = np.linspace(0.1, 1.0, 19, endpoint=True)


def brute_force_search(
    pre_digs: Iterable[PreDig],
    t_begs: Iterable[int],
    t_ends: Iterable[Optional[int]],
    prim: bool = True,
    bio: bool = True,
) -> tuple[pd.Series, pd.DataFrame, dict[str, pd.Series]]:
    """
    Grid search to jointly calibrate a sequence of PreDig objects

    Args:
        pre_digs: an Iterable of PreDig
        t_begs: an Iterable of int specifying first day to consider for calibration
            for each PreDig
        t_ends: an Iterable of int specifying last day to consider for calibration
            for each PreDig
        prim: boolean, specifying if the pre_digs in PreDig have primary sludge input flow
        bio: boolean, specifying if the pre_digs in PreDig have biological sludge input flow
    Return:
        - the calibrated parameter, as a pandas.Series
        - the result of each parameter evaluation, as a pandas.DataFrame
        - a dictionnary specifying confidence intervals on the calibrated parameter (keys: "CI_low", "CI_high")

    The grid search is performed with:
        31 equally spaced points for B0_prim between 200 and 500
        26 equally spaced points for B0_bio between 100 and 350
        19 equally spaced points for k_dis, between 0.1 and 10

    """
    if (not prim) and (not bio):
        raise ValueError("There should be either primary sludge or biological sludge")

    if prim:
        _B0_prim_s = B0_prim_s.copy()
    else:
        _B0_prim_s = np.array([350.0])

    if bio:
        _B0_bio_s = B0_bio_s.copy()
    else:
        _B0_bio_s = np.array([250.0])

    params_to_eval = [
        {"k_dis": k_dis, "B0_prim": B0_prim, "B0_bio": B0_bio}
        for k_dis in k_dis_s
        for B0_prim in _B0_prim_s
        for B0_bio in _B0_bio_s
    ]

    def score(x: dict[str, float]) -> float:
        score_accu = 0.0
        n_data_accu = 0
        for pre_dig, t_beg, t_end in zip(pre_digs, t_begs, t_ends):
            n_data = pre_dig.n_obs(t_beg=t_beg, t_end=t_end)
            score_accu += n_data * pre_dig.score(**x, t_beg=t_beg, t_end=t_end) ** 2
            n_data_accu += n_data
        res = np.sqrt(score_accu / n_data_accu)
        print(f"{x}: {res}")
        return res

    grid_res_flat = np.array(par_eval(score, params_to_eval, True))
    param_opt = params_to_eval[grid_res_flat.argmin()]

    grid = pd.DataFrame(params_to_eval)
    grid["rmse"] = grid_res_flat

    param_opt = grid.iloc[grid["rmse"].argmin()].drop("rmse")

    n_param = 1 + bio + prim
    n_data = sum(
        dig.n_obs(t_beg, t_end) for dig, t_beg, t_end in zip(pre_digs, t_begs, t_ends)
    )

    beale_ratio = np.sqrt(
        1 + n_param / (n_data - n_param) * scs.f(n_param, n_data - n_param).ppf(0.95)
    )
    best_score = grid["rmse"].min()
    thresh_score = best_score * beale_ratio

    grid_uq = grid[grid["rmse"] < thresh_score].drop("rmse")

    return param_opt, grid, {"CI_low": grid_uq.min(), "CI_high": grid_uq.max()}
