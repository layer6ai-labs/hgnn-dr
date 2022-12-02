import argparse
import baseline_utils as bu
import numpy as np
import os
import pickle
from omegaconf import OmegaConf

__DIR__ = os.path.dirname(os.path.realpath(__file__))

_BASE_CONF = OmegaConf.load(os.path.join(__DIR__, "config.yaml"))
_DATA_DIR = _BASE_CONF.data_directory.format(__dir__=__DIR__)

_DATASET_CONF = _BASE_CONF.dataset
parser = argparse.ArgumentParser()
parser.add_argument('--network_split', default=_DATASET_CONF.network_split, type=int)
parser.add_argument('--random_state', default=_DATASET_CONF.random_state, type=int)
args = parser.parse_args()
NETWORK_SPLIT = args.network_split
RANDOM_STATE = args.random_state

OUTDIR = f"{_DATA_DIR}/{_DATASET_CONF.directory}"


# Load multiorg data
print("Loading GCN multi-organism data")
data = bu.loadGCNData(split_seed=NETWORK_SPLIT)

# Load features
with open(f"{OUTDIR}/features.pkl", 'rb') as f:
    features = pickle.load(f)

p_dim = features["protein"]['feature_matrix_pca'].shape[1]
r_dim = features["rdkit"]['feature_matrix'].shape[1]
m_dim = features["mordred"]['feature_matrix_pca'].shape[1]
n_features = sum([p_dim, r_dim, m_dim])

# Build ML-ready dataset
print("Building dataset")

ml = {
    "feature_names": (
        [f"unirep_{i}" for i in range(p_dim)] +
        [f"rdkit_{n}" for n in features["rdkit"]["feature_names"]] +
        [f"mordred_{i}" for i in range(m_dim)]
    ),

    # this is used to filter out other organisms later
    "idx_to_protein": np.array([[i, p] for i, p in data.data["idx_to_protein"].items()]),
    "idx_to_chem": np.array([[i, c] for i, c in data.data["idx_to_chem"].items()])
}

for edgeset in ['other', 'warm', 'test']:
    edges = data.data[f"{edgeset}_edges"].numpy()
    edges_false = data.data[f"{edgeset}_edges_false"].numpy()

    X, y, E = bu.buildDataset(edges, edges_false, data.data, features, n_features,
                              shuffle=RANDOM_STATE)

    ml[f"X_{edgeset}"] = X
    ml[f"y_{edgeset}"] = y
    ml[f"edges_{edgeset}"] = E

for fold, (edges, edges_false) in enumerate(zip(data.data["fold_edges"],
                                                data.data["fold_edges_false"])):
    edgeset = f"fold_{fold}"
    edges, edges_false = edges.numpy(), edges_false.numpy()

    X, y, E = bu.buildDataset(edges, edges_false, data.data, features, n_features,
                              shuffle=RANDOM_STATE)

    ml[f"X_{edgeset}"] = X
    ml[f"y_{edgeset}"] = y
    ml[f"edges_{edgeset}"] = E

edges = data.data[f"test_edges"].numpy()
edges_false = data.data[f"test_edges_false"].numpy()

X, y, E = bu.buildDataset(edges, edges_false, data.data, features, n_features,
                          shuffle=None, neg_to_pos_ratio=None)

ml[f"X_alltest"] = X
ml[f"y_alltest"] = y
ml[f"edges_alltest"] = E

# Save features and dataset
os.makedirs(OUTDIR, exist_ok=True)

output_filename = f'{OUTDIR}/dataset.npz'
np.savez_compressed(output_filename, **ml)

print("Finished building ML-ready baseline dataset with sucess!")
