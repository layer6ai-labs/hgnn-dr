import os
import pickle
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import Sequence

import baseline_utils as bu

__DIR__ = os.path.dirname(os.path.realpath(__file__))

_BASE_CONF = OmegaConf.load(os.path.join(__DIR__, "config.yaml"))
_DATA_DIR = _BASE_CONF.data_directory.format(__dir__=__DIR__)
_DATASET_CONF = _BASE_CONF.dataset

impute_median = lambda: ("imputer", SimpleImputer(strategy="median"))
apply_pca = lambda: ("pca", PCA(n_components=_DATASET_CONF.pca_n_components,
                                whiten=True,
                                random_state=_DATASET_CONF.random_state))

def build_processed_features(
    unirep_paths: Sequence[str] = [
        os.path.join(_DATA_DIR, "data_prep", f"{org}.unirep_features.npz")
        for org in [5664, 5691, 5833]
    ],
    rdkit_path: str = os.path.join(_DATA_DIR, "data_prep", "rdkit_features.tsv.gz"),
    mordred_path: str = os.path.join(_DATA_DIR, "data_prep", "mordred_features.tsv.gz"),
) -> None:

    print("Loading protein (UniRep) features")
    protein_features = bu.loadProteinFeatures(unirep_paths,
                                              transforms=[impute_median, apply_pca],
                                              transform_suffix="_pca")

    print("Loading chemical (RDKit) features")
    rdkit_features = bu.loadChemicalFeatures(rdkit_path,
                                             transforms=[impute_median],
                                             transform_suffix="")

    print("Loading chemical (Mordred) features")
    mordred_features = bu.loadChemicalFeatures(mordred_path,
                                               transforms=[impute_median, apply_pca],
                                               transform_suffix="_pca")

    features = {
        "protein": protein_features,
        "rdkit": rdkit_features,
        "mordred": mordred_features
    }

    print("Saving processed features")
    OUTDIR = f"{_DATA_DIR}/{_DATASET_CONF.directory}"
    os.makedirs(OUTDIR, exist_ok=True)

    with open(f"{OUTDIR}/features.pkl", 'wb') as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    build_baseline_features()
