from copy import deepcopy
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from vsr.pyadm1.eval_vsr.read_data import PreDig


def analyze_vsr(
    pre_digs: Iterable[PreDig],
    t_begs: Iterable[int],
    t_ends: Iterable[Optional[int]],
    k_dis: float,
    B0_prim: float,
    B0_bio: float,
):
    """Analyze VSR for more than one PreDig - deal with case when there
    are multiple non contiguous periods of time
    """

    analysis_res = [
        predig.analyze_vsr(
            k_dis=k_dis, B0_prim=B0_prim, B0_bio=B0_bio, t_beg=t_beg, t_end=t_end
        )
        for predig, t_beg, t_end in zip(pre_digs, t_begs, t_ends)
    ]

    n_obs_s = np.array(
        [
            predig.n_obs(t_beg, t_end)
            for predig, t_beg, t_end in zip(pre_digs, t_begs, t_ends)
        ]
    )

    n_obs_frac = n_obs_s / np.sum(n_obs_s)

    rmse_s = np.array([r["RMSE"] for r in analysis_res])
    mae_s = np.array([r["MAE"] for r in analysis_res])
    bias_s = np.array([r["bias"] for r in analysis_res])

    res = {
        "RMSE": np.sqrt(np.sum(rmse_s**2 * n_obs_frac)),
        "MAE": np.sum(mae_s * n_obs_frac),
        "bias": np.sum(bias_s * n_obs_frac),
    }

    return res


def pred_vsr(
    k_dis: float,
    B0_prim: float,
    B0_bio: float,
    pre_digs: Iterable[PreDig],
):
    accu = []
    for predig in pre_digs:
        idx = deepcopy(predig.pre_feed.index[1:])
        idx.freq = None
        temp = deepcopy(
            pd.Series(
                predig.simulate(k_dis=k_dis, B0_prim=B0_prim, B0_bio=B0_bio)
                .df["VSR"]
                .to_numpy(),
                idx,
            )
        )
        accu.append(temp)
    return pd.concat(accu)
