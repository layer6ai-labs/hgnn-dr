import os
import pickle
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import Sequence

import baseline_utils as bu

from config import __DIR__, DATASET_CONF, DATA_DIR, OUT_DIR, RANDOM_STATE

impute_median = lambda: ("imputer", SimpleImputer(strategy="median"))
apply_pca = lambda: ("pca", PCA(n_components=DATASET_CONF.pca_n_components,
                                whiten=True,
                                random_state=RANDOM_STATE))

def build_processed_features(
    unirep_paths: Sequence[str] = [
        os.path.join(DATA_DIR, "data_prep", f"{org}.unirep_features.npz")
        for org in [5664, 5691, 5833]
    ],
    rdkit_path: str = os.path.join(DATA_DIR, "data_prep", "rdkit_features.tsv.gz"),
    mordred_path: str = os.path.join(DATA_DIR, "data_prep", "mordred_features.tsv.gz"),
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
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(f"{OUT_DIR}/features.pkl", 'wb') as f:
        pickle.dump(features, f)


if __name__ == "__main__":
    build_processed_features()
