import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Type
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model._base import LinearModel
from vsr.empirical_model.feature_engineering import make_new_features

MAX_INTEGER = 4294967295


def LOPO_hyperparam_eval(
    X: pd.DataFrame,
    Y: pd.Series,
    weights: pd.Series,
    hyperparams_to_eval: List[dict],
    model: LinearModel,
    random_state=None,
) -> np.ndarray:
    """Leave one plant out hyperparameter evaluation

    Args:
        X: DataFrame with a MultiIndex (first level, the plants, second level, the dates)
        Y: variable to regress
        weights: weights given to each row
        hyperparms_to_eval: list of hyperparameters to eval
        model: model to fit (should be inherited from sklearn LinearModel, have
            a fit and predict method, and have hyperparameters coherent with hyperparams_to_eval).
    Output:
        list of LOPO cross validation score for each hyperparameter evaluated
    The cross validation score is computed as:
        sqrt( sum_plant sum_{d in plant} w_d (Pred_plant(x_d) - y_d)^2)
    """
    random.seed(random_state)

    # Get list of plant names
    plants_names = list(set(X.index.get_level_values(0)))
    plants_names.sort()

    # Prepare result accu
    res = np.zeros(len(hyperparams_to_eval))

    # Loop on hyperparameters in hyperparams_to_eval
    for i, hyperparam in enumerate(hyperparams_to_eval):

        # Local hyperparameter score accu
        accu = 0.0

        # Loop on plant
        for plant in plants_names:

            # Split train/test with LOPO strategy (train = all but plant, test = plant)
            p_comp = [p for p in plants_names if p != plant]

            X_train, X_test = X.loc[p_comp], X.loc[plant]
            Y_train, Y_test = Y.loc[p_comp], Y.loc[plant]
            weights_train, weights_test = weights.loc[p_comp], weights.loc[plant]

            # Normalize data based on train data
            mean_, std_ = X_train.mean(0), X_train.std(0)
            std_[np.abs(std_) < 1e-14] = 1.0
            X_train_normed = (X_train - mean_) / std_

            # Define/fit model
            ml = model(**hyperparam, random_state=random.randint(0, MAX_INTEGER))
            ml.fit(X_train_normed, Y_train, sample_weight=weights_train)

            # Predict on the test data
            pred = ml.predict((X_test - mean_) / std_)

            # Compute residual
            resi = pred - Y_test.to_numpy()

            # Add weighted MSE to accu
            accu += np.sum((resi**2) * weights_test.to_numpy())

        # MSE -> RMSE
        accu = np.sqrt(accu / np.sum(weights))

        # Store result of hyperparameter evaluation
        res[i] = accu
    return res


def model_to_str(
    model: LinearModel,
    columns: List[str],
    normalizer_mean: Iterable[float],
    normalizer_std: Iterable[float],
) -> str:
    """Transform Linear model to equation string
    Args:
        model: a fitted LinearModel, with coef_ and intercept_ attributes
        columns: names of each column
        normalizer_mean: if LinearModel was trained with normalized inputs,
            this is the mean of the unnormalized inputs
        normalizer_std: if LinearModel was trained with normalized inputs,
        this is the standard dev of the unnormalized inputs

    Returns:
        a string detailing the equation
    """

    # Prepare accu
    str_ = ""

    # Prepare intercept accu
    intercept = model.intercept_

    # Convert to array
    coeffs = np.asarray(model.coef_)
    _cols = np.asarray(columns)
    _means = np.asarray(normalizer_mean)
    _stds = np.asarray(normalizer_std)

    # Remove 0 values
    keep_idx = coeffs != 0.0
    coeffs = coeffs[keep_idx]
    _cols = _cols[keep_idx]
    _means = _means[keep_idx]
    _stds = _stds[keep_idx]

    # Sort by more import
    _order = np.argsort(np.abs(coeffs))[::-1]
    coeffs = coeffs[_order]
    _cols = _cols[_order]
    _means = _means[_order]
    _stds = _stds[_order]

    # Loop on element
    for coeff, name, to_ad, to_div in zip(coeffs, _cols, _means, _stds):

        # element =  coef * (column - to_ad) / to_div
        # => -coef * to_ad / to_div + (coef / to_div) * column
        intercept += -coeff * to_ad / to_div
        if coeff > 0.0:
            str_ += f" + {coeff/to_div} x {name}"
        else:
            str_ += f" - {-coeff/to_div} x {name}"

    return f"VSR = {intercept}{str_}"


@dataclass
class LOPOresult:
    """Class for storing the result of a two level LOPO procedure"""

    model: LinearModel
    hyperparam: dict
    grid_eval: np.ndarray
    residual: np.ndarray
    mean_: np.ndarray
    std_: np.ndarray
    cols: List[str]

    @property
    def MAE(self) -> float:
        return np.mean(np.abs(self.residual))

    @property
    def RMSE(self) -> float:
        return np.sqrt(np.mean(self.residual**2))

    @property
    def bias(self) -> float:
        return np.mean(self.residual)

    def model_str(self) -> str:
        return model_to_str(
            model=self.model,
            columns=self.cols,
            normalizer_mean=self.mean_,
            normalizer_std=self.std_,
        )

    @property
    def small_dict(self):
        return {
            "MAE": self.MAE,
            "RMSE": self.RMSE,
            "bias": self.bias,
            "model_str": self.model_str(),
        }


class LOPOception:
    """
    Train VSR model, using a two stage LOPO procedure to evaluate performance
    of hyperparameter selection methodology.

    The model constructed is a linear model with engineered featured. The linear model is trained
    using (by default) Lasso algorithm, with "alpha" hyperparameter selected by LOPO cross validation
    (Geometric grid of size 20 between 0.01 and 100.0)

    Usage:
    - The training is performed by calling the 'fit' method.
    - After fitting, new predictions can be performed using the 'predict' method.

    Fitting strategy:

    Details of steps:
    1. Feature engineering
    From collection of initial features F1, ..., Fd,
    - constructs extra features Fi ** k for k in [-2, -1, 1, 2]
    - constructs extra features Fi * Fj, Fi / Fj

    2. Split train/test
    Sets aside specified plants and data

    3. Data normalisation
    The train data is used to infer proper data normalisation

    4. Hyperparameter selection
    Selection of Lasso hyperparameter through Leave one plant out cross validation
    This performed, as should be, on the train plants

    5. Proper model training
    Train the model using train data

    6. Test assessment
    Test the model on test data

    Initialisation arguments:
    Args:
        y_col: the name of variable to be regressed in data (default: "VSR_target_mean_[%]")
        feat_engineering_hyperparams: Optional dictionnary specifying how new features should be
            computed. Default is None, amounts to default parameters of 'make_new_features' function.
        model: Type of model to be trained. Default is None, amounts to Lasso. Should be a Type[LinearModel].
        grid_hyperparameters: List of hyperparameters to evaluate. Default is None (Default hyperparameter if model is not Lasso,
            if Lasso considers a Geometric grid for 'alpha' of size 20 between 0.01 and 100.0)
        random_state: random state for reproducibility (used both for LOPO at test + LOPO at train + training)
    """

    def __init__(
        self,
        y_col="VSR_target_mean_[%]",
        feat_engineering_hyperparams: Optional[dict] = None,
        model: Optional[Type[LinearModel]] = None,
        grid_hyperparameters: Optional[List[dict]] = None,
        random_state=None,
    ):

        # Storing main data
        if model is None:
            self.model = Lasso
        else:
            self.model = model

        # Preparing grid of hyperparameters
        if grid_hyperparameters is None:
            if self.model == Lasso:
                self.grid_hyperparameters = [
                    {"alpha": alpha, "fit_intercept": True, "max_iter": 1500}
                    for alpha in np.logspace(-2, 2, 20)
                ]
            else:
                warn(
                    "No grid of hyperparameter was specified. Default model hyperparameter will be used, without hyperparameter selection"
                )
                self.grid_hyperparameters = [{}]
        else:
            self.grid_hyperparameters = grid_hyperparameters

        if feat_engineering_hyperparams is None:
            self.feat_hyperparams = {}
        else:
            self.feat_hyperparams = feat_engineering_hyperparams

        self.y_col = y_col
        self.random_state = random_state

    def _get_weights(self):
        """Compute weights based on number of observations per plant.
        Total plant weight is the square root of the number of observations
        This is a compromise between 'one observation = 1 weight' and
        'one plant = 1 weight' strategy."""
        self.weights_per_plant = {
            name: self.X_y.loc[name].shape[0] for name in self.plants_names
        }
        self.weights = pd.Series(
            np.sqrt([1 / self.weights_per_plant[idx[0]] for idx in self.X_y.index]),
            self.X_y.index,
        )

    def fit(self, X_y: pd.DataFrame):
        """Main fitting task of LOPOception

        Argument:
            X_y, a DataFrame with a two level MultiIndex (first level is plant name, second level date).
                Should contain both X and y value (y in column specified in self.y_col).
                X columns should be compatible with feat_engineering_hyperparams specified at initialisation
        """

        # Store raw data, extract Y data
        self.X_y = X_y.copy()
        del X_y  # Make absolutely sure the raw data is not modified by mistake
        self.Y = self.X_y[self.y_col].copy()

        self.plants_names = list(set(self.X_y.index.get_level_values(0)))
        self.plants_names.sort()

        # 0. Inferring weight per data point
        self._get_weights()

        # 1. Feature engineering
        print("1. Engineering features\n")
        # valid_cols trick: valid columns are not yet known, so set to None for initial
        # call to feature_engineering
        self.valid_cols = None
        self.X_full = self.feature_engineering(self.X_y)
        self.valid_cols = self.X_full.columns

        # 2. Methodology assessment
        print("2. Assessing methodology\n")
        # Do the LOPO LOPO  assessment trick
        self._assess_methodology()
        print(f"Test kpis:\n{self.test_kpis}\n")

        # 3. Hyperparameter selection
        print("3. Selecting hyperparameter\n")
        self._get_train_hyperparameter()
        print(f"Selected hyperparameter: {self.selected_hyperparam}\n")

        # 4. Normalisation
        print("4. Normalising\n")
        self._get_normalizer()
        self.X_normed = self.normalize(self.X_full)

        # 5. Main train!
        print("5. Main training phase\n")
        self._main_train()
        print(f"Constructed model:\n{self.model_str}\n")

        # 6. Model assessment
        print("6. Computing Pipeline Kpis\n")
        self.train_kpis = self._assess_trained_model()
        print(f"Train kpis:\n{self.train_kpis}\n")

        # 7. Over. The model can be called through "predict"
        print("Pipeline over. Make new predictions with 'predict' method")

    def _assess_trained_model(self) -> pd.DataFrame:
        """Assess performance of the trained model on each of the plants
        Returns a DataFrame detailing the train performance (RMSE, MAE, bias)
        for each plant
        """

        # Prepare accus
        rmses = {}
        maes = {}
        bias = {}

        # Loop on all plants
        for plant in self.plants_names:
            pred = self.predict(self.X_y.loc[plant])
            resi = pred - self.Y.loc[plant].to_numpy()

            rmses[plant] = np.sqrt(np.mean(resi**2))
            maes[plant] = np.mean(np.abs(resi))
            bias[plant] = np.mean(resi)

        return pd.DataFrame({"RMSE": rmses, "MAE": maes, "bias": bias})

    def _assess_methodology(self):
        """Assess LOPO hyperparameter selection methodology using a two level
        LOPO approach. For each plant, a train set is defined containing all but
        one plant. LOPO is performed to select an hyperparameter, and then train the
        model on this train set. The performance of the LOPO methodology is then
        computed on the test set containing the left out plant (whose data has not
        been used at all to define the model).
        For each plant, the data stored contains:
            - the trained model ("model" key)
            - the selected hyperparameter ("hyperparam" key)
            - the resulting model equation

        """
        # Copy data to avoid leaks
        X = self.X_full.copy()
        Y = self.Y.copy()

        # Prepare accu
        result = {}

        # Leave one plant out
        for plant in self.plants_names:
            print(f"Assessing methodology for {plant}")

            # Split train/test LOPO way
            plant_complement = [p for p in self.plants_names if p != plant]

            X_train, X_test = X.loc[plant_complement], X.loc[plant]
            Y_train, Y_test = Y.loc[plant_complement], Y.loc[plant]
            weights_train = self.weights[plant_complement]

            # Standardize
            mean_, std_ = X_train.mean(0), X_train.std(0)
            X_train_normed = (X_train - mean_) / std_

            # LOPO inception: evaluate hyperparam grid with LOPO
            res = LOPO_hyperparam_eval(
                X_train_normed,
                Y_train,
                weights_train,
                self.grid_hyperparameters,
                model=self.model,
                random_state=self.random_state,
            )

            # Select best hyperparameter, train
            best_hyperparam = self.grid_hyperparameters[np.argmin(res)]
            ml = self.model(**best_hyperparam, random_state=self.random_state)
            ml.fit(X_train_normed, Y_train, sample_weight=weights_train)

            # Assess on test
            pred = ml.predict((X_test - mean_) / std_)
            residual = pred - Y_test.to_numpy()

            # Store results
            result[plant] = LOPOresult(
                model=ml,
                hyperparam=best_hyperparam,
                grid_eval=res,
                residual=residual,
                cols=self.valid_cols,
                mean_=mean_,
                std_=std_,
            )

        self.assessment_result: dict[str, LOPOresult] = result
        self.test_kpis = pd.DataFrame(
            {key: val.small_dict for key, val in result.items()}
        ).T

    def feature_engineering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Feature engineering function. This is based on make_new_features function
        defined in vsr.empirical_model.feature_engineering. Hyperparameters of make_new_features
        function are stored in "feat_hyperparams" attribute.

        From dataFrame X, returns DataFrame with newly engineered features.
        NOTE: Returned DataFrame and original DataFrame are independant (original data is
        copied)
        """

        # For first run only. Some columns, which have to be rejected in the general case,
        # might be kept in some reduced inputs by the make_new_features
        # For instance, if column X contains [0.0, ..., 0.0, 1.0, ..., 1.0], its inverse
        # would be removed in general case, but if the rows containing 0.0 are filtered,
        # then it would not be removed by make_new_features. This causes difficulties in
        # sklearn, so they are filtered here.
        if self.valid_cols is None:
            return make_new_features(X, **self.feat_hyperparams)

        # General run, with filtering of columns.
        return make_new_features(X, **self.feat_hyperparams)[self.valid_cols]

    def _get_normalizer(self):
        """From full data, get normalizer (i.e. means and std deviations per
        columns which turn columns into centered and reduced columns)."""
        self.normalizer_std = self.X_full.std(0)
        self.normalizer_mean = self.X_full.mean(0)

    def normalize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Normalize a DataFrame using the normalizer information.
        NOTE: the dataframe returned can not assumed to be centered nor reduced.
        But it can be fed to the trained_model, since the normalisation applied is the
        same as was applied to the train data."""
        return (X - self.normalizer_mean) / self.normalizer_std

    def _get_train_hyperparameter(self):
        """Perform LOPO procedure to select best hyperparameter"""

        # Copy data just to be safe
        X = self.X_full.copy()
        Y = self.Y.copy()

        # Evaluate grid of hyperparameter
        self.hyperparams_grid_res = LOPO_hyperparam_eval(
            X,
            Y,
            self.weights.copy(),
            hyperparams_to_eval=self.grid_hyperparameters,
            model=self.model,
            random_state=self.random_state,
        )

        # Store best hyperparameter
        self.selected_hyperparam = self.grid_hyperparameters[
            np.argmin(self.hyperparams_grid_res)
        ]

    def _main_train(self):
        """Train model using all data and hyperparameter selected from LOPO"""
        model = self.model(**self.selected_hyperparam, random_state=self.random_state)
        model.fit(self.normalize(self.X_full), self.Y, sample_weight=self.weights)
        self.trained_model = model

    def _predict(self, X) -> np.ndarray:
        """Predict result from dataframe containing engineered features"""
        X_normed = self.normalize(X)
        return self.trained_model.predict(X_normed)

    def predict(self, X) -> np.ndarray:
        """Predict result from dataframe containing original features"""
        return self._predict(self.feature_engineering(X))

    @property
    def model_str(self) -> str:
        return model_to_str(
            model=self.trained_model,
            columns=self.valid_cols,
            normalizer_mean=self.normalizer_mean,
            normalizer_std=self.normalizer_std,
        )

    def comp_train_kpis_on_dates(self, dates: Dict[str, List[Tuple[str, str]]]):
        perfs = {}
        residuals = self.predict(self.X_y) - self.Y

        def make_time_index(spans):
            dt = pd.DatetimeIndex([])
            for beg, end in spans:
                dt = dt.union(pd.date_range(beg, end))
            return dt

        for plant in dates.keys():
            if plant not in self.plants_names:
                raise ValueError(f"Unknown plant {plant}")
            res = residuals.loc[plant].loc[
                make_time_index(dates[plant]).intersection(self.X_y.loc[plant].index)
            ]
            perfs[plant] = {
                "RMSE": np.sqrt(np.mean(res**2)),
                "MAE": np.mean(np.abs(res)),
                "bias": np.mean(res),
            }

        return pd.DataFrame(perfs).T

    def comp_test_kpis_on_dates(self, dates: Dict[str, List[Tuple[str, str]]]):
        perfs = {}

        def make_time_index(spans: List[Tuple[str, str]]):
            dt = pd.DatetimeIndex([])
            for beg, end in spans:
                dt = dt.union(pd.date_range(beg, end))
            return dt

        for plant in dates:
            info = self.assessment_result[plant]
            idx = self.X_y.loc[plant].index
            residual = pd.Series(info.residual, idx)

            res = residual.loc[make_time_index(dates[plant]).intersection(idx)]
            perfs[plant] = {
                "RMSE": np.sqrt(np.mean(res**2)),
                "MAE": np.mean(np.abs(res)),
                "bias": np.mean(res),
            }
        return pd.DataFrame(perfs).T
