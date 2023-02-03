import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)  # Pandas

import argparse
import json
import numpy as np
import os
import pandas as pd
import pickle
from sklearn.preprocessing import PowerTransformer

from baseline_utils import (evalGlobalAtPrcMetrics, getDatasetSplits,
                            filterEdgesetOrganisms, evalPreds, evalMatrixAtKMetrics)
from models import LogisticRegressionModel, XGBModel, LGBModel, BinaryMLPModel

from config import (__DIR__, DATASET_CONF, MODELS_CONF, OUT_DIR,
                    RANDOM_STATE, MODEL_CONSTANTS, OC_CONVERT)
__DIR__ = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument('--network_split', default=DATASET_CONF.network_split, type=int)
parser.add_argument('--random_state', default=RANDOM_STATE, type=int)
parser.add_argument('--save_metrics_xlsx', default="", type=str)
args = parser.parse_args()
NETWORK_SPLIT = args.network_split
RANDOM_STATE = args.random_state

MODEL_OTHER_ORGANISMS = OC_CONVERT(MODELS_CONF.model_other_organisms)
RETAIN_TRAIN_ORGANISMS = OC_CONVERT(MODELS_CONF.retain_train_organisms)


# ========================
# Evaluate baseline models
# ========================
def evalModels(dataset, models_dict):
    print(f"\n{'='*26}\nEVALUATING BASELINE MODELS\n{'='*26}")

    # Final test performance
    trained_models = dict(
        MODELS_CONF=MODELS_CONF,
        OUT_DIR=OUT_DIR,
        RANDOM_STATE=RANDOM_STATE,
        NETWORK_SPLIT=NETWORK_SPLIT
    )

    train, _, test = getDatasetSplits(dataset, None,
                                      keep_organisms=RETAIN_TRAIN_ORGANISMS,
                                      shuffle=RANDOM_STATE)
    alltest = {k: dataset[f'{k}_alltest'] for k in ['X', 'y', 'edges']}

    power_transform = PowerTransformer()
    trained_models['power_transform'] = power_transform
    train['X_pt'] = power_transform.fit_transform(train['X'])
    test['X_pt'] = power_transform.transform(test['X'])
    alltest['X_pt'] = power_transform.transform(alltest['X'])
    get_X = lambda split, tr: split['X_pt' if tr.transform_X else 'X']

    model_constants = MODEL_CONSTANTS.copy()
    model_constants["feature_names"] = list(dataset['feature_names'])
    del model_constants["early_stopping_rounds"]

    trained_models = dict()
    eval_results = {n: pd.DataFrame() for n in models_dict}
    alltest_scores = edges_and_scores_dataframe(alltest)  # X not needed for this

    for model_name, Model in models_dict.items():
        model_hypers = dict(MODELS_CONF[model_name])
        mdl = Model(model_constants, **model_hypers)

        print(f'\nTraining {model_name} model')
        mdl.fit(get_X(train, mdl), train['y'])
        trained_models[model_name] = mdl.get_model()

        # No negative sampling for final_test metrics
        y_pred_alltest = mdl.predict(get_X(alltest, mdl))
        alltest_scores[f'{model_name}_score'] = y_pred_alltest

        eval_results[model_name] = \
            evalPreds(alltest['y'], y_pred_alltest)
        eval_results[model_name].update(
            evalGlobalAtPrcMetrics(alltest['y'], y_pred_alltest))
        eval_results[model_name].update(
            evalMatrixAtKMetrics(alltest['edges'], alltest['y'], y_pred_alltest))

    print("\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    alltest_scores.to_csv(f"{OUT_DIR}/scores_seed_{RANDOM_STATE}.csv.gz", index=False)

    print(f"Final test performance:")
    for model_name, metrics in eval_results.items():
        print(f"{model_name}:")
        print("  " + ", ".join(["%.4f (%s)" % (metrics[m], m) for m in ["auc", "ap"]]))
        print("  " + ", ".join(["%.4f (%s)" % (metrics[m], m) for m in metrics if m.startswith("p@")]))
        print("  " + ", ".join(["%.4f (%s)" % (metrics[m], m) for m in metrics if m.startswith("r@")]))
        print("  " + ", ".join(["%.4f (%s)" % (metrics[m], m) for m in metrics if m.startswith("prec@")]))
        print("  " + ", ".join(["%.4f (%s)" % (metrics[m], m) for m in metrics if m.startswith("rec@")]))
    print("")

    # Save results
    print('Saving models and their evaluation scores')
    with open(f'{OUT_DIR}/models.pkl', 'wb') as f:
        pickle.dump(trained_models, f)

    with open(f'{OUT_DIR}/evaluation.json', 'w') as f:
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


def write_metrics_xlsx(eval_results, filename, split=NETWORK_SPLIT, seed=RANDOM_STATE):
    all_metrics = {sheet_name: {"split": split, "seed": seed}
                   for sheet_name in ["auc_ap", "global_metrics", "local_metrics"]}

    for model, metrics in eval_results.items():
        all_metrics["auc_ap"] = {**all_metrics["auc_ap"], **{
            f"{model}_test_auc": metrics['auc'],
            f"{model}_test_ap": metrics['ap'],
        }}

        all_metrics["global_metrics"] = {**all_metrics["global_metrics"],
            **{f"{model}_p@{prc}": metrics[f'p@{prc}'] for prc in [0.5, 1]},
            **{f"{model}_r@{prc}": metrics[f'r@{prc}'] for prc in [0.5, 1]},
        }

        all_metrics["local_metrics"] = {**all_metrics["local_metrics"],
            **{f"{model}_prec@{k}": metrics[f'prec@{k}'] for k in[3, 5]},
            **{f"{model}_rec@{k}": metrics[f'rec@{k}'] for k in[3, 5]},
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
    dataset = dict(np.load(f"{OUT_DIR}/dataset.npz"))
    idx_to_protein = {int(r[0]): r[1] for r in dataset["idx_to_protein"]}

    dataset["X_other"], dataset["y_other"], dataset["edges_other"] = \
        filterEdgesetOrganisms(dataset["X_other"], dataset["y_other"], dataset["edges_other"],
                              idx_to_protein, MODEL_OTHER_ORGANISMS)

    eval_results = evalModels(dataset, {
        'lr': LogisticRegressionModel,
        'xgb': XGBModel,
        'lgb': LGBModel,
        'mlp': BinaryMLPModel,
    })

    if len(args.save_metrics_xlsx) > 0:
        sheets = write_metrics_xlsx(eval_results, args.save_metrics_xlsx)
