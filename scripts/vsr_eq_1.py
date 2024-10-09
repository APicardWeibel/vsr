import os
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

loc_dir = os.path.realpath(os.path.dirname(__file__))
save_dir = os.path.join(loc_dir, "EQUATION_1")

# Read data
X_y = pd.read_csv(os.path.join(loc_dir, "all_data.csv"))
X_y["date"] = pd.to_datetime(X_y["date"])
X_y.set_index(["plant", "date"], inplace=True)

# Assessing default model
def equation_1(X: pd.DataFrame) -> pd.Series:
    """Compute VSR according to Eq. 1 model"""
    return 13.7 * np.log(X["HRT_AD[day]"]) + 18.9


test_time_spans = {
    "Dig1": [("2020-12-20", "2022-03-13")],
    "Dig6": [("2021-03-22", "2021-05-01")],
    "Dig3": [("2019-05-18", "2020-07-01")],
    "Dig5": [("2016-02-16", "2016-04-16"), ("2017-03-30", "2017-06-29")],
    "Dig4": [("2016-04-08", "2016-06-30"), ("2021-01-29", "2022-07-01")],
    "Dig2": [("2018-09-23", "2021-12-09")],
}

# Helper to create time index from ranges
def make_time_index(spans):
    dt = pd.DatetimeIndex([])
    for beg, end in spans:
        dt = dt.union(pd.date_range(beg, end))
    return dt


time_indexes = {
    name: make_time_index(test_time_spans[name]) for name in test_time_spans
}

# Overall prediction of VSR
VSR_eq_1 = equation_1(X_y)

pred_test_name = "pred_vsr_eq_1_test"
pred_all_name = "pred_vsr_eq_1_all"
pred_test_path = os.path.join(save_dir, pred_test_name)
pred_all_path = os.path.join(save_dir, pred_all_name)

os.makedirs(pred_test_path, exist_ok=True)
os.makedirs(pred_all_path, exist_ok=True)

plants_names = list(set(X_y.index.get_level_values(0)))
for plant in plants_names:
    print(f"Starting {plant}")
    csv_file = pd.DataFrame([VSR_eq_1.loc[plant], X_y["VSR_[%]"].loc[plant]]).T
    csv_file.columns = ["Eq1", "obs"]
    csv_file.to_csv(os.path.join(pred_all_path, f"{plant}.csv"))

    csv_file = csv_file.loc[time_indexes[plant]]
    csv_file.to_csv(os.path.join(pred_test_path, f"{plant}.csv"))

shutil.make_archive(pred_test_path, "zip", pred_test_path)
shutil.make_archive(pred_all_path, "zip", pred_all_path)


residual = VSR_eq_1 - X_y["VSR_[%]"]


def comp_test_kpis_on_dates(
    residual: pd.Series, dates: Dict[str, List[Tuple[str, str]]]
):
    perfs = {}

    for plant in dates:
        residual_loc = residual.loc[plant]
        idx = residual_loc.index

        res = residual_loc.loc[make_time_index(dates[plant]).intersection(idx)]
        perfs[plant] = {
            "RMSE": np.sqrt(np.nanmean(res**2)),
            "MAE": np.nanmean(np.abs(res)),
            "bias": np.nanmean(res),
        }
    return pd.DataFrame(perfs).T


test_kpis_eq_1 = comp_test_kpis_on_dates(residual, test_time_spans)
print(f"Test kpis:\n{test_kpis_eq_1}")
test_kpis_eq_1.to_csv(os.path.join(save_dir, "test_kpis_eq_1.csv"))
