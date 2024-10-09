from copy import copy
from math import sqrt
from typing import Optional

import numpy as np
import pandas as pd
from vsr.pyadm1.basic_classes.dig_info import ADM1DigInfo
from vsr.pyadm1.basic_classes.feed import ADM1Feed, influent_state_cols
from vsr.pyadm1.basic_classes.param import ADM1Param
from vsr.pyadm1.basic_classes.state import ADM1States, default_ini_state, pred_col
from vsr.pyadm1.digester import Digester
from vsr.pyadm1.eval_vsr.fractionner import (
    ADM1Fractionner,
    biological_sludge_fract,
    primary_sludge_fract,
)
from vsr.pyadm1.model.vs_computation import N_DAY_VSR

n_cols = len(influent_state_cols)

# Define a new initial state from default, setting X_c to 0.0
ini_state = copy(default_ini_state)
ini_state.df["X_c"] = 0.0


def get_ch_pr_li_I(
    vs_flow: np.ndarray, fract: ADM1Fractionner
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute fraction of carbohydrates, proteins, lipids and inert from flow of vs and fractionner"""
    f_cod = vs_flow * fract.cod_vs
    Xch, Xpr, Xli, XI = fract.Xs
    return f_cod * Xch, f_cod * Xpr, f_cod * Xli, f_cod * XI


# Prepare intermediary dataframe for use
inter_cols = ["BS_VS", "BS_Q", "PS_VS", "PS_Q", "V_liq", "VSR"]

temperature = 38 + 273.15


class PreFeed:
    """Information for Feed construction given the B0 information for
    Biological Sludge and Primary Sludge.

    NOTE:
    The volume is renormalized to 1 m3, to account for operational changes
    (when part of the digesters are not used). It is therefore assumed
    that the volume is constant. The flow is modified adequately.
    """

    def __init__(self, data: pd.DataFrame):
        self._V_liq = data["V_liq"].to_numpy()
        self._BS_VS = data["BS_VS"].to_numpy()
        self._BS_Q = data["BS_Q"].to_numpy()
        self._PS_VS = data["PS_VS"].to_numpy()
        self._PS_Q = data["PS_Q"].to_numpy()

        self.index = data.index

        self._Q = self._BS_Q + self._PS_Q

        self._nday = data.shape[0]

        self._prep_feed = pd.DataFrame(
            np.zeros((self._nday, n_cols + 1)),
            index=np.arange(self._nday),
            columns=["V_liq"] + influent_state_cols,
        )
        self._prep_feed["time"] = np.arange(self._nday)
        self._prep_feed["Q"] = self._Q / self._V_liq
        self._prep_feed["V_liq"] = 1.0
        self._prep_feed["T_op"] = temperature

    @property
    def V_liq(self) -> np.ndarray:
        """Liquid volume (in m3). This is the true volume of the digesters."""
        return self._V_liq

    @property
    def BS_VS(self) -> np.ndarray:
        """VS flow from Biological Sludge (in kgVS/d)"""
        return self._BS_VS

    @property
    def BS_Q(self) -> np.ndarray:
        """Biological sludge flow (in m3/d)"""
        return self._BS_Q

    @property
    def PS_VS(self) -> np.ndarray:
        """Primary sludge flow (in kgVS/d)"""
        return self._PS_VS

    @property
    def PS_Q(self) -> np.ndarray:
        """Primary sludge flow (in m3/d)"""
        return self._PS_Q

    @property
    def Q(self) -> np.ndarray:
        """Total sludge flow (in m3/d).
        This is the true flow going inside the digesters
        (i.e. not renormalized for a volume of 1)"""
        return self._Q

    @property
    def prep_feed(self) -> pd.DataFrame:
        """Pre prepared feed (missing X values)"""
        return self._prep_feed.copy()

    def to_adm1(self, Bo_PS: float = 450.0, Bo_BS: float = 250.0) -> ADM1Feed:
        """Return an ADM1 compatible Feed"""
        feed = self.prep_feed
        primary_sludge_fract.Bo = Bo_PS
        biological_sludge_fract.Bo = Bo_BS
        f_bs_ch, f_bs_pr, f_bs_li, f_bs_I = get_ch_pr_li_I(
            self._BS_VS, biological_sludge_fract
        )
        f_ps_ch, f_ps_pr, f_ps_li, f_ps_I = get_ch_pr_li_I(
            self._PS_VS, primary_sludge_fract
        )

        feed["X_ch"] = (f_bs_ch + f_ps_ch) / self._Q  # kgCOD/m3
        feed["X_pr"] = (f_bs_pr + f_ps_pr) / self._Q  # kgCOD/m3
        feed["X_li"] = (f_bs_li + f_ps_li) / self._Q  # kgCOD/m3
        feed["X_I"] = (f_bs_I + f_ps_I) / self._Q

        return ADM1Feed(feed)


def make_intermed_df(df: pd.DataFrame) -> tuple[PreFeed, ADM1States]:
    n_df = pd.DataFrame(
        np.zeros((len(df.index), len(inter_cols))), index=df.index, columns=inter_cols
    )
    n_df["V_liq"] = df["Volume"]
    n_df["BS_VS"] = (
        df["BS_flow_[m3/d]"] * df["TS_BS_[gTS/L]"] * df["VS_BS_[gVS/gTS]"]
    )  # kgVS /D
    n_df["BS_Q"] = df["BS_flow_[m3/d]"]
    n_df["PS_VS"] = (
        df["PS_flow_[m3/d]"] * df["TS_PS_[gTS/L]"] * df["VS_PS_[gVS/gTS]"]
    )  # kgVS /D
    n_df["PS_Q"] = df["PS_flow_[m3/d]"]

    obs_df = pd.DataFrame(
        np.full((len(df.index), len(pred_col)), np.nan),
        index=range(len(df.index)),
        columns=pred_col,
    )
    obs_df["VSR"] = df["VSR"].to_numpy() / 100.0
    return PreFeed(n_df), ADM1States(obs_df)


class PreDig:
    """All informations inferred from excel files prior to knowledge of
    parameters required to compute the feed"""

    def __init__(
        self, pre_feed: PreFeed, V_liq: float = 1.0, obs: Optional[ADM1States] = None
    ):
        self.dig_info = ADM1DigInfo(V_liq=V_liq, V_gas=V_liq * 0.1)
        self.pre_feed = pre_feed

        self.pre_feed._prep_feed[["Q", "V_liq"]] *= V_liq

        self.ini_state = ini_state
        self.obs = obs

    @property
    def obs(self):
        return self._obs

    @obs.setter
    def obs(self, obs: Optional[ADM1States]):
        if obs is None:
            self._obs = None
            self._VSR = None
        else:
            self._obs = obs
            self._VSR = obs.df["VSR"].to_numpy()

    @property
    def VSR(self):
        return self._VSR

    def to_adm1(self, B0_prim: float = 450.0, B0_bio: float = 250.0) -> Digester:
        """Transform PreDigester in digester"""
        return Digester(
            dig_info=self.dig_info,
            feed=self.pre_feed.to_adm1(Bo_PS=B0_prim, Bo_BS=B0_bio),
            ini_state=self.ini_state,
            obs=self.obs,
        )

    def simulate(self, k_dis: float = 0.5, B0_prim: float = 450, B0_bio: float = 250):
        """Simulate using ADM1 model

        ADM1 Parameters considered for this setting are k_dis.
        The two other parameters of interest, B0_prim and B0_bio, are used to compute
        the feed specification
        """

        return self.to_adm1(B0_prim, B0_bio).simulate(
            ADM1Param({"k_hyd_li": k_dis, "k_hyd_pr": k_dis, "k_hyd_ch": k_dis})
        )

    def comp_VSR_res(
        self,
        k_dis: float = 0.5,
        B0_prim: float = 450,
        B0_bio: float = 250,
        t_beg: int = 0,
        t_end: Optional[int] = None,
    ):
        output = self.simulate(k_dis=k_dis, B0_prim=B0_prim, B0_bio=B0_bio)

        residuals = output.df["VSR"].to_numpy() - self._VSR[1:]

        if t_end is None:
            t_end = len(residuals)
        return residuals[t_beg:t_end]

    def n_obs(self, t_beg: int = 0, t_end: Optional[int] = None):
        if t_end is None:
            t_end = len(self._VSR)

        return np.sum(~np.isnan(self._VSR[:-N_DAY_VSR][t_beg:t_end]))

    def score(
        self,
        k_dis: float = 0.5,
        B0_prim: float = 450,
        B0_bio: float = 250,
        t_beg: int = 0,
        t_end: Optional[int] = None,
    ) -> float:
        """Score parameters k_dis, B0_prim and B0_bio.
        Args:
            k_dis, B0_prim, B0_bio: parameters to calibrate
            t_beg, t_end: beginning and end date between which the error is computed
        returns:
            Root mean square error of VSR prediction
        """
        try:
            residuals = self.comp_VSR_res(
                k_dis=k_dis, B0_prim=B0_prim, B0_bio=B0_bio, t_beg=t_beg, t_end=t_end
            )

            return sqrt(np.nanmean(residuals**2))
        except Exception as exc:
            print(
                f"Failed for parameter 'k_dis':{k_dis}, 'B0_prim':{B0_prim}, 'B0_bio':{B0_bio}"
            )
            raise exc
            print(exc)
            return 1.0

    def analyze_vsr(
        self,
        k_dis: float = 0.5,
        B0_prim: float = 450,
        B0_bio: float = 250,
        t_beg: int = 0,
        t_end: Optional[int] = None,
    ) -> dict[str, float]:
        residuals = self.comp_VSR_res(
            k_dis=k_dis, B0_prim=B0_prim, B0_bio=B0_bio, t_beg=t_beg, t_end=t_end
        )
        return {
            "RMSE": sqrt(np.nanmean(residuals**2)),
            "MAE": np.nanmean(np.abs(residuals)),
            "bias": np.nanmean(residuals),
        }


def make_pre_dig(df: pd.DataFrame, V_liq: float = 1.0) -> PreDig:
    pre_feed, obs = make_intermed_df(df)
    return PreDig(pre_feed, V_liq, obs)


def read_csv(
    path: str,
    V_liq: float = 1.0,
    date_beg: Optional[str] = None,
    date_end: Optional[str] = None,
    time_format: Optional[str] = None,
) -> PreDig:

    df = pd.read_csv(path, sep=";", index_col=0)
    df.index = pd.to_datetime(df.index, format=time_format)
    if date_beg is None:
        date_beg = df.index[0]
    else:
        date_beg = pd.Timestamp(date_beg)
    if date_end is None:
        date_end = df.index[-1]
    else:
        date_end = pd.Timestamp(date_end)

    df_red: pd.DataFrame = df.loc[(df.index >= date_beg) & (df.index <= date_end)]

    return make_pre_dig(df_red.resample("1D").interpolate(), V_liq=V_liq)
