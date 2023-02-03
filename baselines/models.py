import warnings
warnings.simplefilter(action='ignore', category=UserWarning)  # LightGBM

import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from typing import Dict

from config import MODEL_CONSTANTS
from neural_nets import BinaryMLP

def pop_dict(d, key, default=None, use_default=True):
    if key in d:
        result = d[key]
        del d[key]
    elif use_default:
        result = default
    else:
        raise KeyError
    return result


class GenericModel:
    def __init__(self, constants:Dict=MODEL_CONSTANTS, **hypers):
        """
        (Global) Constants are used to specify settings across all models: this is to allow values
        not used for this specific model but should be handled gracefully because they may be
        useful for other model.

        Hypers have variable arguments that are meant to be fed directly to some method for the model
        class, and should not contain any unexpected values.
        """

        self.constants = constants.copy()
        self.hypers = hypers.copy()

        # Special flags
        self.verbosity = pop_dict(self.constants, "verbosity", -1)
        self.transform_X = pop_dict(self.hypers, "transform_X", False)

    def fit(self, X, y, X_val=None, y_val=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

    def score(self, X_val, y_val):
        """Score the model. Higher is better."""
        return average_precision_score(y_val, self.predict(X_val))

    def get_model(self):
        raise NotImplementedError

    def get_best_val_iter(self):
        raise NotImplementedError


class LogisticRegressionModel(GenericModel):
    def __init__(self, constants=MODEL_CONSTANTS, **kwargs):
        super().__init__(constants=constants, **kwargs)

        self.max_iter = self.hypers["max_iter"]
        self.model = LogisticRegression(random_state=self.constants["random_state"], **self.hypers)

    def fit(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

    def get_model(self):
        return self.model

    def get_best_val_iter(self):
        # return int(self.model.n_iter_[0])
        return {"max_iter": self.max_iter}


class XGBModel(GenericModel):
    def __init__(self, constants=MODEL_CONSTANTS, **kwargs):
        super().__init__(constants=constants, **kwargs)

        self.hypers["seed"] = self.constants["random_state"]
        self.hypers["objective"] = "binary:logistic"
        self.hypers["eval_metric"] = ["logloss", "aucpr"]
        self.hypers["verbosity"] = min(self.verbosity, 1)

        self.feature_names = self.constants.get("feature_names", None)

        self.train_params = dict(
            num_boost_round = pop_dict(self.hypers, "num_boost_round", use_default=False),
            early_stopping_rounds = self.constants.get("early_stopping_rounds", None),
            verbose_eval = max(self.verbosity, 0),
        )

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Returns a trained XGBoost model and prediction function
        """
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feature_names)

        if X_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val)
            evallist = [(dval, 'val'), (dtrain, 'train')]
        else:
            evallist = [(dtrain, 'train')]

        self.evals_result = dict()
        self.model = xgb.train(self.hypers, dtrain,
                               evals=evallist,
                               evals_result=self.evals_result,
                               **self.train_params)

    def predict(self, X):
       return self.model.predict(xgb.DMatrix(X, feature_names=self.feature_names))

    def get_model(self):
        return self.model

    def get_best_val_iter(self):
        return {"num_boost_round": self.model.best_iteration}


class LGBModel(GenericModel):
    def __init__(self, constants=MODEL_CONSTANTS, **kwargs):
        super().__init__(constants=constants, **kwargs)

        self.hypers["seed"] = self.constants["random_state"]
        self.hypers["objective"] = "binary"
        self.hypers["metric"] = ["binary_logloss", "average_precision"]
        self.hypers["verbosity"] = min(self.verbosity, 1)

        self.feature_names = self.constants.get("feature_names", None)

        self.train_params = dict(
            num_boost_round = pop_dict(self.hypers, "num_boost_round", use_default=False),
            early_stopping_rounds = self.constants.get("early_stopping_rounds", 0),
            verbose_eval = max(self.verbosity, 0),
        )

    def fit(self, X, y, X_val=None, y_val=None):
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
        val_data = None if X_val is None else [lgb.Dataset(X_val, label=y_val)]

        self.model = lgb.train(self.hypers, train_data, valid_sets=val_data, **self.train_params)

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model

    def get_best_val_iter(self):
        return {"num_boost_round": self.model.best_iteration}


class BinaryMLPModel(GenericModel):
    def __init__(self, constants=MODEL_CONSTANTS, **kwargs):
        super().__init__(constants=constants, **kwargs)

        self.model = BinaryMLP(verbosity=self.verbosity,
                               seed=self.constants["random_state"],
                               **self.hypers)

    def fit(self, X, y, X_val=None, y_val=None):
        return self.model.fit(X, y, X_val, y_val)

    def predict(self, X):
        return self.model.predict(X)

    def get_model(self):
        return self.model

    def get_best_val_iter(self):
        return {"max_iter": self.model.best_iter}
