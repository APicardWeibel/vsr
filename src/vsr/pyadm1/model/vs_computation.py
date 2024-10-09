"""
VS computations for both intrant and digester.

Added to original ADM1 implementation by (https://github.com/CaptainFerMag/PyADM1) to compute VSR
"""

import numba as nb
import numpy as np
import vsr.misc.nb_typ as nbt
from vsr.pyadm1.basic_classes.feed import cod_vs_feed_cols, q_col
from vsr.pyadm1.basic_classes.Phy import cod_vs_values
from vsr.pyadm1.basic_classes.state import cod_vs_dig_states_cols

N_DAY_VSR = 30


@nb.njit(nbt.f1D(nbt.f1D, nbt.f1D))
def volume_per_day(t: np.ndarray, v_liq: np.ndarray):
    day_begin = int(t[0]) + 1
    day_end = int(t[-1])

    n_days = day_end - day_begin + 1

    v_liq_d = np.zeros(n_days)

    loc = 0
    for i in range(len(t[1:])):
        ti1 = t[i + 1]
        v = v_liq[i]
        if ti1 >= (loc + day_begin):
            v_liq_d[loc] = v
            loc += 1
    return v_liq_d


@nb.njit(nbt.UTuple(nbt.f1D, 2)(nbt.f2D))
def feed_vs(
    dig_feed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute volatile solid concentration in the feed.

    Args:
        dig_feed: a DigesterFeed object, on which the VS part is computed
        per_day: should the VS be characterized for a day or between t and dt?
        day_begin: which is the first day in the dig_feed file? (See below).
    If per_day is True:
        As usual, the description of the substrate at line i is assumed to be valid
        from t_{i-1} to t_{i}. No information is accessible therefore for the substrate
        before t_0. For the first and last day, VS is computed using the mean information
        for that day.

    Returns Feed Volatile solid in kg VS M-3 and Mass flow in kg/ Day
    """

    # Get vs_in at each time step
    vs_in = dig_feed[:, cod_vs_feed_cols] @ (1 / cod_vs_values)  # kg VS M-3

    # Take the mean of the VS in the day.
    q_in = dig_feed[:, q_col]  # M3 Day-1
    t = dig_feed[:, 0]

    day_begin = int(t[0]) + 1
    day_end = int(t[-1])

    n_days = day_end - day_begin + 1
    vs_per_day = np.zeros(n_days)
    q_per_day = np.zeros(n_days)

    ti = t[0]
    ti1 = t[1]
    vs_accu = 0
    loc_day = 0
    v_accu = 0

    for i in range(len(t[1:])):
        vs_conc = vs_in[i]
        q = q_in[i]
        ti1 = t[i + 1]
        if ti1 >= (loc_day + day_begin):
            # New day: compute mean value, store result, prepare new value

            # Contribution of vs to loc_day
            vs_accu += (loc_day + day_begin - ti) * q * vs_conc  # kg VS
            v_accu += (loc_day + day_begin - ti) * q  # m3

            # Store value
            vs_per_day[loc_day] = vs_accu / v_accu
            q_per_day[loc_day] = v_accu

            # Reset and Contribution of vs to loc_day + 1
            vs_accu = (ti1 - day_begin + loc_day) * q * vs_conc  # kg VS
            v_accu = (ti1 - day_begin + loc_day) * q  # m3
            loc_day += 1

        else:
            # Contribution of vs to loc_day
            vs_accu += (ti1 - ti) * q * vs_conc  # kg VS
            v_accu += (ti1 - ti) * q  # m3
        ti = ti1  # update value of ti

    # End loop

    # Correct for first day (potentially, t[0] is not an integer
    # so renormalisation is wrong)
    t0 = t[0]
    renorm = 1 / (1 + int(t0) - t0)
    vs_per_day[0] = vs_per_day[0] * renorm
    q_per_day[0] = q_per_day[0] * renorm

    return vs_per_day, q_per_day


def dig_states_vs(dig_states: np.ndarray) -> np.ndarray:
    """
    Volatile solid in the digester (and by extension in the digester output).
    In kg VS M-3

    """
    return dig_states[:, cod_vs_dig_states_cols] @ (1 / cod_vs_values)  # kgVS M-3
