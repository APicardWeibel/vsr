import os
import random
import shutil

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from vsr.empirical_model.vsr_model_train import LOPOception

# To exactly reproduce results if the data is not changed
random_state = 42
random.seed(random_state)
np.random.seed(random_state)

# Time span info for test
test_time_spans = {
    "A_Dig1": [("2020-12-20", "2022-03-13")],
    "B_Dig6": [("2021-03-22", "2021-05-01")],
    "C_Dig3": [("2019-05-18", "2020-07-01")],
    "D_Dig5": [("2016-02-16", "2016-04-16"), ("2017-03-30", "2017-06-29")],
    "E_Dig4": [("2016-04-08", "2016-06-30"), ("2021-01-29", "2022-07-01")],
    "F_Dig2": [("2018-09-23", "2021-12-09")],
}

# (Technical) name translation
# This is to ensure that the results are reproduced (it depends upon
# ordering of iteration on dict, which follows alphabetical order)
name_translation = {name: name[2:] for name in test_time_spans}
inverse_translate = {val: key for key, val in name_translation.items()}

loc_dir = os.path.realpath(os.path.dirname(__file__))
save_dir = os.path.join(loc_dir, "EMPIRICAL_MODEL")
os.makedirs(save_dir, exist_ok=True)

# Read data
X_y = pd.read_csv(os.path.join(loc_dir, "all_data.csv"))
X_y["date"] = pd.to_datetime(X_y["date"])
X_y["plant"] = [inverse_translate[a] for a in X_y["plant"]]
X_y.set_index(["plant", "date"], inplace=True)

# Assess LOPO/Train through LOPO
trainer = LOPOception(
    y_col="VSR_[%]",
    model=Lasso,
    grid_hyperparameters=[
        {
            "alpha": alpha,
            "fit_intercept": True,
            "selection": "random",
            "max_iter": 10000,
        }
        for alpha in np.logspace(-2, 2, 40)
    ],
    random_state=random_state,
)

trainer.fit(X_y)
print(f"Selected hyperparam: {trainer.selected_hyperparam}")

print(f"Trained model with all data: {trainer.model_str}")
trainer.train_kpis.rename(index=name_translation).to_csv(
    os.path.join(save_dir, "train_kpis.csv")
)
trainer.test_kpis.rename(index=name_translation).to_csv(
    os.path.join(save_dir, "test_kpis.csv")
)

train_test_kpis = trainer.comp_train_kpis_on_dates(test_time_spans)
test_test_kpis = trainer.comp_test_kpis_on_dates(test_time_spans)

print(f"KPIs of global model on test dates:\n{train_test_kpis}\n\n")
print(f"KPIs of LOPO trained model on test dates:\n{test_test_kpis}\n\n")

train_test_kpis.rename(index=name_translation).to_csv(
    os.path.join(save_dir, "test_kpis_global_emp.csv")
)
test_test_kpis.rename(index=name_translation).to_csv(
    os.path.join(save_dir, "test_kpis_lopo_emp_test.csv")
)

print("Recomputing VSR predictions")
name_pred_test = "pred_vsr_emp_model"
name_pred_all = "pred_vsr_emp_model_all"
path_pred_test = os.path.join(save_dir, name_pred_test)
path_pred_all = os.path.join(save_dir, name_pred_all)
os.makedirs(path_pred_test, exist_ok=True)
os.makedirs(path_pred_all, exist_ok=True)


def make_time_index(spans):
    dt = pd.DatetimeIndex([])
    for beg, end in spans:
        dt = dt.union(pd.date_range(beg, end))
    return dt


for plant, time_span_list in test_time_spans.items():
    info = trainer.assessment_result[plant]
    idx = trainer.X_y.loc[plant].index
    residual = pd.Series(info.residual, idx)

    true_vsr = trainer.X_y.loc[plant][trainer.y_col]
    pred_vsr = true_vsr + residual

    index = make_time_index(time_span_list).intersection(idx)

    result_all = pd.DataFrame(
        [true_vsr, pred_vsr],
        columns=idx,
        index=["obs", "Emp_model"],
    ).T
    result_test = pd.DataFrame(
        [true_vsr.loc[index], pred_vsr.loc[index]],
        columns=index,
        index=["obs", "Emp_model"],
    ).T
    result_all.to_csv(os.path.join(path_pred_all, f"{name_translation[plant]}.csv"))
    result_test.to_csv(os.path.join(path_pred_test, f"{name_translation[plant]}.csv"))

shutil.make_archive(path_pred_test, "zip", path_pred_test)
shutil.make_archive(path_pred_all, "zip", path_pred_all)
