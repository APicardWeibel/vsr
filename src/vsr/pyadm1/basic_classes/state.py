from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from vsr.pyadm1.basic_classes.Phy import COD_VS

pred_col = [
    "time",
    "S_su",
    "S_aa",
    "S_fa",
    "S_va",
    "S_bu",
    "S_pro",
    "S_ac",
    "S_h2",
    "S_ch4",
    "S_IC",
    "S_IN",
    "S_I",
    "X_c",
    "X_ch",
    "X_pr",
    "X_li",
    "X_su",
    "X_aa",
    "X_fa",
    "X_c4",
    "X_pro",
    "X_ac",
    "X_h2",
    "X_I",
    "S_cation",
    "S_anion",
    "pH",
    "S_va_ion",
    "S_bu_ion",
    "S_pro_ion",
    "S_ac_ion",
    "S_hco3_ion",
    "S_nh3",
    "S_gas_h2",
    "S_gas_ch4",
    "S_gas_co2",
    "S_co2",
    "S_nh4_ion",
    "q_gas",
    "q_ch4",
    "p_ch4",
    "p_co2",
    "VS",
    "VS_in",
    "VSR",
]

predict_units_dict = {
    "time": "Day",
    "S_su": "kgCOD M-3",
    "S_aa": "kgCOD M-3",
    "S_fa": "kgCOD M-3",
    "S_va": "kgCOD M-3",
    "S_bu": "kgCOD M-3",
    "S_pro": "kgCOD M-3",
    "S_ac": "kgCOD M-3",
    "S_h2": "kgCOD M-3",
    "S_ch4": "kgCOD M-3",
    "S_IC": "kmole C M-3",
    "S_IN": "kmole N M-3",
    "S_I": "kgCOD M-3",
    "X_c": "kgCOD M-3",
    "X_ch": "kgCOD M-3",
    "X_pr": "kgCOD M-3",
    "X_li": "kgCOD M-3",
    "X_su": "kgCOD M-3",
    "X_aa": "kgCOD M-3",
    "X_fa": "kgCOD M-3",
    "X_c4": "kgCOD M-3",
    "X_pro": "kgCOD M-3",
    "X_ac": "kgCOD M-3",
    "X_h2": "kgCOD M-3",
    "X_I": "kgCOD M-3",
    "S_cation": "kmole M-3",
    "S_anion": "kmole M-3",
    "pH": "pH",
    "S_va_ion": "kgCOD M-3",
    "S_bu_ion": "kgCOD M-3",
    "S_pro_ion": "kgCOD M-3",
    "S_ac_ion": "kgCOD M-3",
    "S_hco3_ion": "kmole M-3",
    "S_nh3": "kmole M-3",
    "S_gas_h2": "kgCOD M-3",
    "S_gas_ch4": "kgCOD M-3",
    "S_gas_co2": "kmole M-3",
    "S_co2": "kmole M-3",
    "S_nh4_ion": "kmole M-3",
    "q_gas": "M3 Day-1",
    "q_ch4": "M3 Day-1",
    "p_ch4": "bar",
    "p_co2": "bar",
    "VS": "kgVS M-3",
    "VS_in": "kgVS M-3",
    "VSR": "ratio",
}

pred_col_dict = {name: i for i, name in enumerate(pred_col)}
cod_vs_dig_states_cols = np.array([pred_col_dict[name] for name in COD_VS.keys()])


class ADM1States:
    def __init__(self, df: pd.DataFrame):
        _df = deepcopy(df)
        self.df = _df

    @property
    def np_data(self):
        return self._df.to_numpy()

    @property
    def df(self):
        """Panda dataframe representation of the data"""
        return self._df

    @df.setter
    def df(self, value: pd.DataFrame):
        # Check col names
        missing_cols = set(value.columns).difference(pred_col)
        extra_cols = set(pred_col).difference(value.columns)
        if len(extra_cols) > 0:
            raise ValueError(f"Could not interpret the following columns: {extra_cols}")

        if not "time" in value.columns:
            raise ValueError("Observation should have a 'time' column")

        time = value["time"].to_numpy()
        if (len(time) > 1) and np.any(time[1:] - time[:-1] <= 0):
            raise ValueError("Time information should be increasing")

        # Fill missing columns
        if len(missing_cols):
            value[missing_cols] = np.NaN

        self._df = value[pred_col]

    def save(self, path):
        """Save Observations object to a .csv file"""
        self._df.to_csv(path, index=False)
        return path

    def __repr__(self):
        return self._df.__repr__()

    def __str__(self):
        return self._df.__str__()

    def split(self, time_split: float):
        """
        Returns a tuple containing the feed information up to time and the feed information after time.
        Split is done so that the first information can give prediction up to time included.
        """
        time_feed = self._df["time"]
        cond = time_feed < time_split

        obs_before = self._df.iloc[cond]
        obs_after = self._df.iloc[~cond]
        return (ADM1States(obs_before), ADM1States(obs_after))

    def plot(self, pred_name: str, *args, **kwargs):
        data = self._df[pred_name]
        name_with_unit = f"{pred_name} (in {predict_units_dict[pred_name]})"
        if "label" in kwargs.keys():
            plt.plot(self._df["time"], data, *args, **kwargs)
        else:
            plt.plot(self._df["time"], data, *args, label=name_with_unit, **kwargs)
        return plt

    def get_state(self, index):
        return ADM1State(self._df.iloc[index])


def load_states(path: str) -> ADM1States:
    return ADM1States(pd.read_csv(path))


class ADM1State:
    def __init__(self, state: pd.Series):
        self.df = state.copy()

    @property
    def np_data(self) -> np.ndarray:
        return self._df.to_numpy()

    @property
    def df(self) -> pd.Series:
        """Panda dataframe representation of the data"""
        return self._df

    @df.setter
    def df(self, value: pd.Series):
        # Check if all data is present
        missing = set(pred_col).difference(value.index)
        extra = set(value.index).difference(pred_col)

        err_msg = []
        if missing:
            err_msg.append(f"Missing indexes: {missing}")

        if extra:
            err_msg.append(f"Unknown indexes: {missing}")

        err_str = "\nFurthermore: ".join(err_msg)
        if err_str:
            raise ValueError(err_str)

        self._df = value[pred_col]

    def save(self, path: str) -> str:
        """Save Observations object to a .csv file"""
        self._df.to_json(path, orient="index")
        return path


def load_state(path: str) -> ADM1State:
    return ADM1State(
        pd.read_json(
            path,
            typ="series",
            orient="index",
        )
    )


# From Rosen and Jeppsson
default_ini_state = ADM1State(
    pd.Series(
        {
            "time": 0.0,  # Suppose relative time
            "S_su": 0.0119548297170,
            "S_aa": 0.0053147401716,
            "S_fa": 0.0986214009308,
            "S_va": 0.0116250064639,
            "S_bu": 0.0132507296663,
            "S_pro": 0.0157836662845,
            "S_ac": 0.1976297169375,
            "S_h2": 0.0000002359451,
            "S_ch4": 0.0550887764460,
            "S_IC": 0.1526778706263,
            "S_IN": 0.1302298158037,
            "S_I": 0.3286976637215,
            "X_c": 0.3086976637215,
            "X_ch": 0.0279472404350,
            "X_pr": 0.1025741061067,
            "X_li": 0.0294830497073,
            "X_su": 0.4201659824546,
            "X_aa": 1.1791717989237,
            "X_fa": 0.2430353447194,
            "X_c4": 0.4319211056360,
            "X_pro": 0.1373059089340,
            "X_ac": 0.7605626583132,
            "X_h2": 0.3170229533613,
            "X_I": 25.6173953274430,
            "S_cation": 0.0400000000000,
            "S_anion": 0.0200000000000,
            "pH": 7.4655377698929,
            "S_va_ion": 0.0115962470726,
            "S_bu_ion": 0.0132208262485,
            "S_pro_ion": 0.0157427831916,
            "S_ac_ion": 0.1972411554365,
            "S_hco3_ion": 0.1427774793921,
            "S_nh3": 0.0040909284584,
            "S_gas_h2": 0.0000102410356,
            "S_gas_ch4": 1.6256072099814,
            "S_gas_co2": 0.0141505346784,
            "S_co2": 0.0099003912343,
            "S_nh4_ion": 0.1261388873452,
            "q_gas": 2955.70345419378,
            "q_ch4": 1799.33,
            "p_ch4": 0.6507796328232,
            "p_co2": 0.3625527133281,
            "VS": np.nan,
            "VS_in": np.nan,
            "VSR": np.nan,
        }
    )
)
