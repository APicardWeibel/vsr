# pylint: disable=all
import os

import pandas as pd
import vsr.basic_io as io
import vsr.pyadm1.eval_vsr.read_data as rd
from vsr.misc import tprint
from vsr.pyadm1.eval_vsr.analyze_vsr import analyze_vsr, pred_vsr
from vsr.pyadm1.eval_vsr.grid_search import brute_force_search
from vsr.pyadm1.model.vs_computation import N_DAY_VSR

loc_dir = os.path.realpath(os.path.dirname(__file__))
save_path = os.path.join(loc_dir, "ADM1_MODEL")
os.makedirs(save_path, exist_ok=True)

FRAC_TRAIN = 0.75

V_liqs = {
    "Dig1": 13300,
    "Dig2": 111300,
    "Dig3": 120000,
    "Dig4": 10000,
    "Dig5": 60100,
    "Dig6": 4000,
}

got_prim = {
    "Dig1": False,
    "Dig2": True,
    "Dig3": True,
    "Dig4": False,
    "Dig5": True,
    "Dig6": False,
}

got_bio = {
    "Dig1": True,
    "Dig2": True,
    "Dig3": True,
    "Dig4": True,
    "Dig5": False,
    "Dig6": True,
}

dates = {
    "Dig1": ({"date_beg": None, "date_end": None},),
    "Dig2": ({"date_beg": None, "date_end": None},),
    "Dig3": ({"date_beg": None, "date_end": "2020-07-31"},),
    "Dig4": (
        {"date_beg": None, "date_end": "2016-07-30"},
        {"date_beg": "2016-10-26", "date_end": None},
    ),
    "Dig5": (
        {"date_beg": None, "date_end": "2016-05-16"},
        {"date_beg": "2016-06-29", "date_end": "2017-07-29"},
    ),
    "Dig6": ({"date_beg": None, "date_end": None},),
}


default_param = {"B0_prim": 450.0, "B0_bio": 250.0, "k_dis": 0.5}

accu_cal_pars = {}
accu_cal_pars_low = {}
accu_cal_pars_high = {}

accu_def_vsr_train = {}
accu_def_vsr_test = {}
accu_def_vsr_all = {}

accu_cal_vsr_train = {}
accu_cal_vsr_test = {}
accu_cal_vsr_all = {}

accu_pred_vsr_def = {}
accu_pred_vsr_cal = {}

accu_vsr_pred_cal = {}
accu_vsr_pred_def = {}

# Loop on all Digesters
for name in V_liqs:
    print(f"Starting {name}")

    predigs = [
        rd.read_csv(
            os.path.join(loc_dir, "data", f"{name}.csv"), V_liqs[name], **date_dict
        )
        for date_dict in dates[name]
    ]
    t_begs = [0 for _ in dates[name]]
    t_ends = [
        int((predig.pre_feed.prep_feed.shape[0] - N_DAY_VSR) * FRAC_TRAIN)
        for predig in predigs
    ]

    opt_param_std, grid_res, param_uq = brute_force_search(
        predigs,
        t_begs=t_begs,
        t_ends=t_ends,
        prim=got_prim[name],
        bio=got_bio[name],
    )

    accu_cal_pars[name] = opt_param_std
    accu_cal_pars_low[name] = param_uq["CI_low"]
    accu_cal_pars_high[name] = param_uq["CI_high"]

    io.rw_jsonlike.save(os.path.join(save_path, f"{name}_grid_res.json"), grid_res)
    io.rw_jsonlike.save(os.path.join(save_path, f"{name}_cal_par.json"), opt_param_std)
    io.rw_jsonlike.save(
        os.path.join(save_path, f"{name}_cal_par_lb.json"), param_uq["CI_low"]
    )
    io.rw_jsonlike.save(
        os.path.join(save_path, f"{name}_cal_par_ub.json"), param_uq["CI_high"]
    )

    # Calibrated param perf
    cal_res_train = analyze_vsr(
        pre_digs=predigs, t_begs=t_begs, t_ends=t_ends, **opt_param_std
    )
    cal_res_test = analyze_vsr(
        pre_digs=predigs,
        t_begs=t_ends,
        t_ends=[None for _ in predigs],
        **opt_param_std,
    )
    cal_res_tot = analyze_vsr(
        pre_digs=predigs,
        t_begs=t_begs,
        t_ends=[None for _ in predigs],
        **opt_param_std,
    )
    print(f"Calibrated, train: {cal_res_train}")
    print(f"Calibrated, test: {cal_res_test}")
    print(f"Calibrated, all: {cal_res_tot}")

    accu_cal_vsr_train[name] = cal_res_train
    accu_cal_vsr_test[name] = cal_res_test
    accu_cal_vsr_all[name] = cal_res_tot

    accu_vsr_pred_cal[name] = pred_vsr(**opt_param_std, pre_digs=predigs)

    # Default param perf
    def_res_train = analyze_vsr(
        pre_digs=predigs, t_begs=t_begs, t_ends=t_ends, **default_param
    )
    def_res_test = analyze_vsr(
        pre_digs=predigs,
        t_begs=t_ends,
        t_ends=[None for _ in predigs],
        **default_param,
    )
    def_res_tot = analyze_vsr(
        pre_digs=predigs,
        t_begs=t_begs,
        t_ends=[None for _ in predigs],
        **default_param,
    )

    print(f"Default, train: {def_res_train}")
    print(f"Default, test: {def_res_test}")
    print(f"Default, all: {def_res_tot}")

    accu_def_vsr_train[name] = def_res_train
    accu_def_vsr_test[name] = def_res_test
    accu_def_vsr_all[name] = def_res_tot

    accu_vsr_pred_def[name] = pred_vsr(**default_param, pre_digs=predigs)


tprint("Completed computations")
# Convert to pandas
df_cal_pars = pd.DataFrame(accu_cal_pars)
df_cal_pars_low = pd.DataFrame(accu_cal_pars_low)
df_cal_pars_high = pd.DataFrame(accu_cal_pars_high)

df_def_vsr_train = pd.DataFrame(accu_def_vsr_train)
df_def_vsr_test = pd.DataFrame(accu_def_vsr_test)
df_def_vsr_all = pd.DataFrame(accu_def_vsr_all)

df_cal_vsr_train = pd.DataFrame(accu_cal_vsr_train)
df_cal_vsr_test = pd.DataFrame(accu_cal_vsr_test)
df_cal_vsr_all = pd.DataFrame(accu_cal_vsr_all)

print(f"df_cal_pars:\n{df_cal_pars}\n")

print(f"df_def_vsr_train:\n{df_def_vsr_train}\n")
print(f"df_def_vsr_test:\n{df_def_vsr_test}\n")
print(f"df_def_vsr_all:\n{df_def_vsr_all}\n")

print(f"df_cal_vsr_train:\n{df_cal_vsr_train}\n")
print(f"df_cal_vsr_test:\n{df_cal_vsr_test}\n")
print(f"df_cal_vsr_all:\n{df_cal_vsr_all}\n")

df_cal_pars.to_csv(os.path.join(save_path, "df_cal_pars.csv"))
df_cal_pars_low.to_csv(os.path.join(save_path, "df_cal_pars_low.csv"))
df_cal_pars_high.to_csv(os.path.join(save_path, "df_cal_pars_high.csv"))

df_def_vsr_train.to_csv(os.path.join(save_path, "df_def_vsr_train.csv"))
df_def_vsr_test.to_csv(os.path.join(save_path, "df_def_vsr_test.csv"))
df_def_vsr_all.to_csv(os.path.join(save_path, "df_def_vsr_all.csv"))

df_cal_vsr_train.to_csv(os.path.join(save_path, "df_cal_vsr_train.csv"))
df_cal_vsr_test.to_csv(os.path.join(save_path, "df_cal_vsr_test.csv"))
df_cal_vsr_all.to_csv(os.path.join(save_path, "df_cal_vsr_all.csv"))

for name in accu_vsr_pred_cal:
    print("Saving time series")
    accu_vsr_pred_cal[name].to_csv(os.path.join(save_path, f"pred_vsr_cal_{name}.csv"))
    accu_vsr_pred_def[name].to_csv(os.path.join(save_path, f"pred_vsr_def_{name}.csv"))
