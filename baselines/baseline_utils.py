"""Baseline model utils for preparing ML-ready datasets"""

import numpy as np
import os
import pandas as pd
import sys
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.sparse import csr_matrix

from config import DATASET_CONF, RANDOM_STATE

MISSINGS_THRESHOLD = DATASET_CONF.missings_threshold
ZEROS_THRESHOLD = DATASET_CONF.zeros_threshold
PCA_N_COMPONENTS = DATASET_CONF.pca_n_components
NEG_TO_POS_RATIO = DATASET_CONF.neg_to_pos_ratio
CLEAN_ARRAYS = DATASET_CONF.clean_arrays


def loadGCNData(split_seed=DATASET_CONF.network_split):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))

    from gcn.config import parser
    from gcn.utils.data_utils import Data

    args, _ = parser.parse_known_args()
    args.split_seed = split_seed
    args.skip_feats = True
    data = Data(args=args).build()

    return data

def loadProteinFeatures(unirep_filenames, transforms=None,
                                         fit_transforms=True,
                                         transform_suffix=""):
    """
    Loads a .npz file of protein UniRep representations
    Returns a dictionary with the Full UniRep matrix, a
    PCA-reduced UniRep matrix, and a map of protein-id to
    row index in the UniRep matrices
    """

    proteins, features = [], []
    for fname in unirep_filenames:
        unirep_data = np.load(fname)
        proteins.append(unirep_data['proteins'])
        features.append(unirep_data['H_avg'])

    proteins = np.concatenate(proteins)
    protein_to_idx = {proteins[i]:i for i in range(len(proteins))}
    features = np.concatenate(features, axis=0)

    features_dict = {'feature_matrix':features, 'protein_to_index':protein_to_idx}

    if transforms is not None:
        transforms = Pipeline([t() if callable(t) else t for t in transforms])
        transformed = transforms.fit_transform(features) if fit_transforms else transforms.transform(features)
        features_dict.update({f'feature_matrix{transform_suffix}':transformed, 'transforms': transforms})

    return features_dict

def loadChemicalFeatures(filename, missings_threshold=MISSINGS_THRESHOLD,
                                  zeros_threshold=ZEROS_THRESHOLD,
                                  transforms=None,
                                  fit_transforms=True,
                                  fit_transform_remove_drugs=None,
                                  transform_suffix=""):
    """
    Loads a .tsv file of chemical descriptors.
    Returns a dictionary with the feature matrix after
    filtering out features with too many missings or zeros,
    and a map from chemical-id to row index for the matrix.
    """
    chem_data = pd.read_csv(filename, sep='\t').set_index("chemical")

    zeros_fraction = (chem_data==0).sum(axis=0)/chem_data.shape[0]
    missings_fraction = chem_data.isnull().sum(axis=0)/chem_data.shape[0]

    remove_features = list(zeros_fraction[zeros_fraction >= zeros_threshold].index)
    remove_features += list(missings_fraction[missings_fraction >= missings_threshold].index)
    chem_data.drop(remove_features, axis=1, inplace=True)

    chemical_to_idx = {c:i for i, c in enumerate(chem_data.index)}
    features = chem_data.values

    features_dict = {'feature_names': list(chem_data.columns),
                     'feature_matrix': features,
                     'chemical_to_index':chemical_to_idx}

    if transforms is not None:
        transforms = Pipeline([t() if callable(t) else t for t in transforms])

        if fit_transforms:
            if fit_transform_remove_drugs is not None:
                fit_features = chem_data[~chem_data.index.isin(fit_transform_remove_drugs)].values
                transforms.fit(fit_features)
                transformed = transforms.transform(features)
            else:
                transformed = transforms.fit_transform(features)
        else:
            transformed = transforms.transform(features)

        features_dict.update({f'feature_matrix{transform_suffix}': transformed, 'transforms': transforms})

    return features_dict

def buildDataset(edges, edges_false, data, features, n_features,
                 neg_to_pos_ratio=NEG_TO_POS_RATIO,
                 clean_arrays_flag=CLEAN_ARRAYS,
                 shuffle=None):
    """
    Builds and saves Numpy (compressed) arrays to train and evaluate
    ML models. The <dataset> argument should be one of "train", "val",
    or "test"
    """

    n_true_edges, n_false_edges = len(edges), len(edges_false)

    if shuffle is not None:
        np.random.seed(shuffle)
        edges_false_sampled = edges_false[np.random.permutation(n_false_edges)]
    else:
        edges_false_sampled = edges_false

    if neg_to_pos_ratio is not None:
        edges_false_sampled = edges_false_sampled[:int(len(edges)*neg_to_pos_ratio)]

    comb_edges = np.concatenate([edges, edges_false_sampled])
    n_comb_edges = len(comb_edges)

    X = np.zeros([n_comb_edges, n_features])
    y = np.zeros(n_comb_edges)
    y[0:n_true_edges] = 1

    for i, edge in enumerate(comb_edges):
        X[i] = extractFeatureVectorFromEdge(edge, data, features)

    if clean_arrays_flag:
        X, y, comb_edges, kept_ratio = cleanArrays(X, y, comb_edges)
        print(f"Cleaned arrays kept {kept_ratio}% of original rows")

    else:
        print("Arrays were not cleaned and all rows were kept")

    order = np.random.permutation(len(y))
    return X[order], y[order], comb_edges[order]

def extractFeatureVectorFromEdge(edge, data, features):
    """
    Given a node pair (edge) in the network, the function creates a concatenated
    vector of [UniRep]-[RDKit] features. If a protein or chemical is not found
    in the feature matrices, the function returns np.nan as a "null" value
    """

    protein_features = features["protein"]
    rdkit_features = features["rdkit"]
    mordred_features = features["mordred"]

    try:
        prot_node, chem_node = edge[0], edge[1]
        protein_id = data['idx_to_protein'][int(prot_node)]
        chemical_id = data['idx_to_chem'][int(chem_node)]

        row_idx_X_rdkit = rdkit_features['chemical_to_index'][chemical_id]
        rdkit_vector = rdkit_features['feature_matrix'][row_idx_X_rdkit]

        row_idx_X_mordred = mordred_features['chemical_to_index'][chemical_id]
        mordred_vector = mordred_features['feature_matrix_pca'][row_idx_X_mordred]

        row_idx_X_protein = protein_features['protein_to_index'][protein_id]
        prot_vector = protein_features['feature_matrix_pca'][row_idx_X_protein]

    except KeyError:
        return np.nan

    return np.concatenate([prot_vector, rdkit_vector, mordred_vector])

def cleanArrays(X, y, edges, imputer=SimpleImputer(strategy="median"), remove_null_rows=False):
    """
    Imputes NaN values and removes rows with all "null_value"
    across columns
    """
    new_X = imputer.fit_transform(X) if imputer is not None else X.copy()

    if remove_null_rows:
        rows_to_keep = np.count_nonzero(~np.isnan(X), axis=1) > 0
        kept_ratio = round(100.0 * sum(rows_to_keep)/new_X.shape[0], 2)
        new_X = new_X[rows_to_keep,:]
        new_y = y[rows_to_keep].copy()
        new_edges = edges[rows_to_keep]
    else:
        new_y = y
        new_edges = edges
        kept_ratio = 100.00

    return new_X, new_y, new_edges, kept_ratio

def getDatasetFolds(dataset):
    folds = map(lambda s: (s, s.find("fold_")), dataset.keys())
    folds = sorted(list(set(s[i+5:] for (s, i) in folds if i>0)))
    return folds

def getDatasetSplits(dataset, fold='0', keep_organisms="all", shuffle=None):
    if shuffle is not None:
        np.random.seed(shuffle)

    if keep_organisms!="all":
        keep_protein_idxs = np.array([
            int(i) for i, p in zip(*dataset["idx_to_protein"].T)
            if int(p[:4]) in keep_organisms])

    def get(splits, keep_organisms="all"):
        if type(splits) == str:
            data = {"X": dataset[f"X_{splits}"],
                    "y": dataset[f"y_{splits}"],
                    "edges": dataset[f"edges_{splits}"]}
        else:
            data = {"X": [], "y":[], "edges":[]}

            def appendto(array_to, array_from):
                if array_to is not None:
                    array_to.append(array_from)

            for split in splits:
                appendto(data["X"], dataset[f"X_{split}"])
                appendto(data["y"], dataset[f"y_{split}"])
                appendto(data["edges"], dataset[f"edges_{split}"])
            data = {k: np.concatenate(d) for (k, d) in data.items()}

        if shuffle is not None:
            order = np.random.permutation(len(data["y"]))
            data = {k: d[order] for (k, d) in data.items()}

        if keep_organisms != "all":
            keep_rows = np.isin(data["edges"][:,0], keep_protein_idxs)
            data = {k: d[keep_rows] for (k, d) in data.items()}

        return data

    folds = getDatasetFolds(dataset)
    train_splits = [] if dataset["edges_other"] is None else ["other"]

    if fold is None:
        train_splits += ["warm"] + [f"fold_{f}" for f in folds]
        val = None
    else:
        train_splits += ["warm"] + sorted([f"fold_{f}" for f in folds if f!=fold])
        val = get(f"fold_{fold}", keep_organisms)

    train = get(train_splits, keep_organisms)
    test = get("test", "all")

    return train, val, test

def filterEdgesetOrganisms(X, y, edges, idx_to_protein, keep_organisms):
    if len(keep_organisms) == 0:
        return None, None, None
    elif keep_organisms == "all":
        return X, y, edges

    check_organism = np.vectorize(lambda pidx: int(idx_to_protein[pidx].split(".")[0]) in keep_organisms)
    keep_rows = check_organism(edges[:,0])

    if len(keep_rows) == 0:
        return None, None, None
    return X[keep_rows], y[keep_rows], edges[keep_rows]


# ======================
# Evaluation and metrics
# ======================

sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/..")
from gcn.utils.eval_utils import precision_recall_at_k

def evalPreds(y_true, y_pred):
    return dict(
        auc = roc_auc_score(y_true, y_pred),
        ap = average_precision_score(y_true, y_pred)
    )

def evalGlobalAtPrcMetrics(y_true, y_pred):
    n = len(y_true)
    npos = (y_true==1).sum()
    pred_sort = np.argsort(y_pred)[::-1]

    metrics = dict()
    for prc in [0.5, 1]:
        ntop = round(n * prc/100)
        ntop_pos = (y_true[pred_sort[:ntop]]==1).sum()

        metrics[f"p@{prc}"] = ntop_pos/ntop
        metrics[f"r@{prc}"] = ntop_pos/npos

    return metrics

def evalMatrixAtKMetrics(edges, y_true, y_pred):
    df = pd.DataFrame(edges, columns=("p", "c"))
    df["true"] = y_true
    df["pred"] = y_pred

    valid_p = df.groupby("p")["true"].sum() > 0
    valid_p = set(valid_p.index[valid_p])
    df = df[df["p"].isin(valid_p)]

    # for each protein in test/val, generate the scores for all the chemicals in test/val
    protein_idx = {item: i for i, item in enumerate(df["p"].unique())}
    chem_idx = {item: i for i, item in enumerate(df["c"].unique())}

    rows = df["p"].apply(protein_idx.get).values
    cols = df["c"].apply(chem_idx.get).values

    gt_matrix = csr_matrix((df["true"].values, (rows, cols))).toarray()
    probs_matrix = csr_matrix((df["pred"].values, (rows, cols))).toarray()

    at_k_metrics={}
    for topk in [3, 5]:
        at_k_metrics[f'prec@{topk}'],at_k_metrics[f'rec@{topk}'] = precision_recall_at_k(gt_matrix, probs_matrix, k=topk)

    return at_k_metrics

def at_k(metric, truth, pred, k, pred_argsort=None):
    k = min(len(truth), k)
    args_topk = np.argsort(pred)[-k:] if pred_argsort is None else pred_argsort[-k:]
    hits_topk = truth[args_topk].sum()

    if metric=="precision":
        return hits_topk/k
    elif metric=="recall":
        return hits_topk/truth.sum()
    elif metric=="ap":
        return (hits_topk.cumsum()/np.arange(1,k+1)).mean()
