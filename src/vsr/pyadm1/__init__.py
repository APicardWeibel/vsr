"""
PyADM1 - Anaerobic digestion modelisation with Anerobic Digestion Model 1 (ADM1)

Details about the modelisation can be found at https://doi.org/10.2166/wst.2002.0292 .
This package is based around ADM1 implementation https://github.com/CaptainFerMag/PyADM1 .

PyADM1 is designed for:

    - Modelisation (run_adm1)
    - Sensitivity Analysis (submodule SA)
    - Calibration (submodule optim)
    - Uncertainty Quantification (submodule UQ)

Permanent data storage is organised around basic_classes.
Anaerobic Digestion objects (DigesterInfo, DigesterFeed, DigesterParameter, ADM1State,
ADM1States) are stored as human readable files (csv, json) and can be loaded/saved using
load_(dig_feed/dig_info/dig_param/dig_state/dig_states) and save methods.

Main functions:
    run_adm1 (ADM1 modeling of digester from initial state, digester information, digester
        parameter and influent)
    adm1_err (measures the difference between predictions and observations)
    score_param (measures the difference between predictions using a specific parameters and
        observations)
"""

from vsr.pyadm1.basic_classes import (
    ADM1DigInfo,
    ADM1Feed,
    ADM1Param,
    ADM1State,
    ADM1States,
    load_dig_info,
    load_feed,
    load_param,
    load_state,
    load_states,
)
from vsr.pyadm1.digester import Digester
from vsr.pyadm1.prediction_error import adm1_err

# from .var_buq import adm1_var_bayes_uq
