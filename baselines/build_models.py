import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Pandas
warnings.simplefilter(action='ignore', category=UserWarning)  # LightGBM

import argparse
import json
import lightgbm as lgb
import numpy as np
import os
import pandas as pd
import pickle
import xgboost as xgb
from baseline_utils import (evalGlobalAtPrcMetrics, getDatasetFolds, getDatasetSplits,
                            filterEdgesetOrganisms, evalPreds, evalMatrixAtKMetrics)
from neural_nets import BinaryMLP
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PowerTransformer

__DIR__ = os.path.dirname(os.path.realpath(__file__))

_BASE_CONF = OmegaConf.load(f"{__DIR__}/config.yaml")
_DATA_DIR = _BASE_CONF.data_directory.format(__dir__=__DIR__)
BASELINES_DIR = f"{_DATA_DIR}/baselines"

_DATASET_CONF = _BASE_CONF.dataset
_MODELS_CONF = _BASE_CONF.models
MAX_ITER = _MODELS_CONF.max_iter
EARLY_STOPPING_ROUNDS = _MODELS_CONF.early_stopping_rounds
VERBOSE_EVAL = _MODELS_CONF.verbose_eval

parser = argparse.ArgumentParser()
parser.add_argument('--network_split', default=4567, type=int)
parser.add_argument('--random_state', default=_DATASET_CONF.random_state, type=int)
parser.add_argument('--save_metrics_xlsx', default="", type=str)
args = parser.parse_args()
NETWORK_SPLIT = args.network_split
SEED = args.random_state

def _OC_CONVERT(config):
    try:
        return OmegaConf.to_object(config)
    except ValueError:
        return config

MODEL_OTHER_ORGANISMS = _OC_CONVERT(_MODELS_CONF.model_other_organisms)
RETAIN_TRAIN_ORGANISMS = _OC_CONVERT(_MODELS_CONF.retain_train_organisms)


# ==============
# Model trainers
# ==============
class LogisticRegressionTrainer:
    def __init__(self,
        max_iter = MAX_ITER,
        random_state = SEED,
        transform_X = _MODELS_CONF.logistic_regression.transform_X,
        hypers = _OC_CONVERT(_MODELS_CONF.logistic_regression.hypers),
        **kwargs
    ):
        self.model = LogisticRegression(max_iter=max_iter,
                                        random_state=random_state,
                                        **hypers)
        self.transform_X = transform_X

    def train(self, X, y, X_val=None, y_val=None):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:,1]

    def get_model(self):
        return self.model

    def get_best_iter(self):
        return int(self.model.n_iter_[0])

class XGBTrainer:
    def __init__(self,
        max_iter = MAX_ITER,
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        seed = SEED,
        transform_X = _MODELS_CONF.xgb.transform_X,
        hypers = _OC_CONVERT(_MODELS_CONF.xgb.hypers),
        verbose_eval = VERBOSE_EVAL,
        feature_names = None,
        **kwargs
    ):
        self.hypers = hypers
        self.hypers.update({"seed": seed})

        self.max_iter = max_iter
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.feature_names = feature_names
        self.transform_X = transform_X

    def train(self, X, y, X_val=None, y_val=None):
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
        self.bst = xgb.train(self.hypers, dtrain, self.max_iter, evallist,
                             early_stopping_rounds=self.early_stopping_rounds,
                             evals_result=self.evals_result,
                             verbose_eval=self.verbose_eval)

    def predict(self, X):
       return self.bst.predict(xgb.DMatrix(X, feature_names=self.feature_names))

    def get_model(self):
        return self.bst

    def get_best_iter(self):
        return self.bst.best_iteration+1


class LGBTrainer:
    def __init__(self,
        max_iter = MAX_ITER,
        early_stopping_rounds = EARLY_STOPPING_ROUNDS,
        seed = SEED,
        transform_X = _MODELS_CONF.lgb.transform_X,
        hypers = _OC_CONVERT(_MODELS_CONF.lgb.hypers),
        verbose_eval = VERBOSE_EVAL,
        feature_names = None,
        **kwargs
    ):
        self.hypers = hypers
        self.hypers.update({"seed": seed})

        self.max_iter = max_iter
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.feature_names = None if feature_names is None else list(feature_names)
        self.transform_X = transform_X

    def train(self, X, y, X_val=None, y_val=None):
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_names)
        val_data = None if X_val is None else [lgb.Dataset(X_val, label=y_val)]

        self.bst = lgb.train(self.hypers, train_data, self.max_iter,
                             valid_sets=val_data,
                             early_stopping_rounds=self.early_stopping_rounds,
                             verbose_eval=self.verbose_eval)

    def predict(self, X):
        return self.bst.predict(X, num_iteration=self.bst.best_iteration)

    def get_model(self):
        return self.bst

    def get_best_iter(self):
        return self.bst.best_iteration+1


# ========================
# Evaluate baseline models
# ========================
def evalModels(dataset, trainers):
    model_names = trainers().keys()
    models = dict()
    eval_results = {model_name: {'val': pd.DataFrame(), 'test': pd.DataFrame()} for model_name in model_names}
    best_iters = {model_name: [] for model_name in model_names}
    feature_names = list(dataset['feature_names'])

    print(f"\n{'='*26}\nEVALUATING BASELINE MODELS\n{'='*26}")
    folds = getDatasetFolds(dataset)
    for fold_num, fold in enumerate(folds):
        print(f"\nFold {fold}:\n{'='*7}")
        fold_models, fold_results = dict(), dict()

        train, val, test = getDatasetSplits(dataset, fold,
                                            keep_organisms=RETAIN_TRAIN_ORGANISMS,
                                            shuffle=SEED)
        power_transform = PowerTransformer()
        train['X_pt'] = power_transform.fit_transform(train['X'])
        val['X_pt'] = power_transform.transform(val['X'])
        test['X_pt'] = power_transform.transform(test['X'])
        get_X = lambda split, tr: split['X_pt' if tr.transform_X else 'X']

        kwdict = {model_name: {'fold': fold_num} for model_name in model_names}
        for model_name, tr in trainers(kwdict).items():
            print(f'\nTraining {model_name} model')
            tr.train(get_X(train, tr), train['y'], get_X(val, tr), val['y'])
            fold_models[model_name] = tr.get_model()

            val_metrics = evalPreds(val['y'], tr.predict(get_X(val, tr)))
            test_metrics = evalPreds(test['y'], tr.predict(get_X(test, tr)))

            fold_results[model_name] = {'val':val_metrics, 'test':test_metrics}
            best_iters[model_name].append(tr.get_best_iter())

        print("")
        for model_name, metrics in fold_results.items():
            print("%s: %.4f (val auc), %.4f (test auc); %.4f (val ap), %.4f (test ap)" % (
                model_name,
                metrics['val']['auc'], metrics['test']['auc'],
                metrics['val']['ap'], metrics['test']['ap']
            ))
        print("")

        models[fold] = fold_models
        for m, d in eval_results.items():
            for s, df in d.items():
                row = pd.Series(fold_results[m][s], name=fold)
                eval_results[m][s] = df.append(row)

    # Save avg / std across folds
    for m, d in eval_results.items():
        for s, df in d.items():
            mean = df.mean().rename("mean")
            std = df.std().rename("std")
            eval_results[m][s] = eval_results[m][s].append([mean, std]).to_dict()

    # Final test performance
    models['final'] = dict()
    kwdict = {
        model_name: {'fold': 'final', 'max_iter': max(b), 'early_stopping_rounds': None, 'feature_names': feature_names}
        for model_name, b in best_iters.items()
    }
    print('Max iterations for final models:\n{}\n{}'.format({m: d['max_iter'] for m, d in kwdict.items()}, '='*50))

    train, _, test = getDatasetSplits(dataset, None,
                                      keep_organisms=RETAIN_TRAIN_ORGANISMS,
                                      shuffle=SEED)
    alltest = {k: dataset[f'{k}_alltest'] for k in ['X', 'y', 'edges']}
    alltest_scores = edges_and_scores_dataframe(alltest)  # X not needed for this

    power_transform = PowerTransformer()
    models['final']['power_transform'] = power_transform
    train['X_pt'] = power_transform.fit_transform(train['X'])
    test['X_pt'] = power_transform.transform(test['X'])
    alltest['X_pt'] = power_transform.transform(alltest['X'])
    get_X = lambda split, tr: split['X_pt' if tr.transform_X else 'X']

    for model_name, tr in trainers(kwdict).items():
        print(f'\nTraining final {model_name} model')
        tr.train(get_X(train, tr), train['y'])
        models['final'][model_name] = tr.get_model()

        # No negative sampling for final_test metrics
        y_pred_alltest = tr.predict(get_X(alltest, tr))
        alltest_scores[f'{model_name}_score'] = y_pred_alltest
        eval_results[model_name]['final_test'] = evalPreds(alltest['y'], y_pred_alltest)
        eval_results[model_name]['final_test'].update(
            evalGlobalAtPrcMetrics(alltest['y'], y_pred_alltest))
        eval_results[model_name]['final_test'].update(
            evalMatrixAtKMetrics(alltest['edges'], alltest['y'], y_pred_alltest))
    print("\n")

    os.makedirs(BASELINES_DIR, exist_ok=True)
    alltest_scores.to_csv(f"{BASELINES_DIR}/scores_seed_{SEED}.csv.gz", index=False)

    print(f"Average across {len(folds)} folds:")
    for model_name, metrics in eval_results.items():
        print("%s: %.4f±%.4f (val auc), %.4f±%.4f (val ap)" % (
            model_name,
            metrics['val']['auc']['mean'], metrics['val']['auc']['std'],
            metrics['val']['ap']['mean'], metrics['val']['ap']['std']
        ))
    print("")

    print(f"Final test performance:")
    for model_name, metrics in eval_results.items():
        final_metrics = metrics['final_test']
        print(f"{model_name}:")
        print("  " + ", ".join(["%.4f (%s)" % (final_metrics[m], m) for m in ["auc", "ap"]]))
        print("  " + ", ".join(["%.4f (%s)" % (final_metrics[m], m) for m in final_metrics if m.startswith("p@")]))
        print("  " + ", ".join(["%.4f (%s)" % (final_metrics[m], m) for m in final_metrics if m.startswith("r@")]))
        print("  " + ", ".join(["%.4f (%s)" % (final_metrics[m], m) for m in final_metrics if m.startswith("prec@")]))
        print("  " + ", ".join(["%.4f (%s)" % (final_metrics[m], m) for m in final_metrics if m.startswith("rec@")]))
    print("")

    # Save results
    print('Saving models and their evaluation scores')
    with open(f'{BASELINES_DIR}/models.pkl', 'wb') as f:
        pickle.dump(models, f)

    with open(f'{BASELINES_DIR}/evaluation.json', 'w') as f:
        json.dump(eval_results, f, indent=4)

    print('Finished with success')
    return eval_results


def edges_and_scores_dataframe(data, split=None):
    df = pd.DataFrame(data['edges'], columns=('protein', 'chem'))
    idx_to_protein = {int(i): p for (i, p) in dataset['idx_to_protein']}
    idx_to_chem = {i: c for (i, c) in dataset['idx_to_chem']}
    df['protein'] = df['protein'].apply(idx_to_protein.get)
    df['chem'] = df['chem'].apply(idx_to_chem.get)
    df['truth'] = data['y']

    if split is not None:
        df.insert(0, "split", split)

    return df


def write_metrics_xlsx(eval_results, filename, split=NETWORK_SPLIT, seed=SEED):
    all_metrics = {sheet_name: {"split": split, "seed": seed}
                   for sheet_name in ["auc_ap", "global_metrics", "local_metrics"]}

    for model, metrics in eval_results.items():
        all_metrics["auc_ap"] = {**all_metrics["auc_ap"], **{
            f"{model}_valid_auc": metrics['val']['auc']['mean'],
            f"{model}_valid_ap": metrics['val']['ap']['mean'],
            f"{model}_test_auc": metrics['final_test']['auc'],
            f"{model}_test_ap": metrics['final_test']['ap'],
        }}

        all_metrics["global_metrics"] = {**all_metrics["global_metrics"],
            **{f"{model}_p@{prc}": metrics['final_test'][f'p@{prc}'] for prc in [0.5, 1]},
            **{f"{model}_r@{prc}": metrics['final_test'][f'r@{prc}'] for prc in [0.5, 1]},
        }

        all_metrics["local_metrics"] = {**all_metrics["local_metrics"],
            **{f"{model}_prec@{k}": metrics['final_test'][f'prec@{k}'] for k in[3, 5]},
            **{f"{model}_rec@{k}": metrics['final_test'][f'rec@{k}'] for k in[3, 5]},
        }

    sheets = dict()
    for sheet_name, metrics in all_metrics.items():
        try:
            df = pd.read_excel(filename, sheet_name=sheet_name)
        except (FileNotFoundError, ValueError):
            df = pd.DataFrame(columns=metrics.keys())

        df = df.append(metrics, ignore_index=True)
        sheets[sheet_name] = df

    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    for sheet_name, df in sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    writer.save()

    return sheets



if __name__ == "__main__":
    import torch
    if _MODELS_CONF.torch_64:
        torch.set_default_dtype(torch.float64)

    mlp_params = dict(max_iter = MAX_ITER,
                      early_stopping_rounds = EARLY_STOPPING_ROUNDS,
                      seed = SEED,
                      transform_X = _MODELS_CONF.mlp.transform_X,
                      verbosity = VERBOSE_EVAL,
                      **_OC_CONVERT(_MODELS_CONF.mlp.hypers))

    trainers = lambda kwdict={}: {
        'lr': LogisticRegressionTrainer(**kwdict.get('lr', {})),
        'xgb': XGBTrainer(**kwdict.get('xgb', {})),
        'lgb': LGBTrainer(**kwdict.get('lgb', {})),
        'mlp': BinaryMLP(**{**mlp_params, **kwdict.get('mlp', {})}),
    }

    dataset = dict(np.load(f"{_DATA_DIR}/{_DATASET_CONF.directory}/dataset.npz"))
    idx_to_protein = {int(r[0]): r[1] for r in dataset["idx_to_protein"]}

    dataset["X_other"], dataset["y_other"], dataset["edges_other"] = \
        filterEdgesetOrganisms(dataset["X_other"], dataset["y_other"], dataset["edges_other"],
                              idx_to_protein, MODEL_OTHER_ORGANISMS)

    eval_results = evalModels(dataset, trainers)

    if len(args.save_metrics_xlsx) > 0:
        sheets = write_metrics_xlsx(eval_results, args.save_metrics_xlsx)
