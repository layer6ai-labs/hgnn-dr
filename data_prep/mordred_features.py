import pandas as pd
import warnings
from mordred import Calculator, descriptors, is_missing
from omegaconf import DictConfig, OmegaConf
from os import path
from rdkit import Chem
from tqdm import tqdm
from typing import Optional, Union, Sequence

from .utils import get_delim, load_drugs_file


__DIR__ = path.dirname(path.realpath(__file__))


def create_mordred_features_file(
    drugs: Optional[Sequence[Union[str, int]]] = None,
    config: Union[DictConfig, str] = path.join(__DIR__, "config.yaml"),
    input_file: Optional[str] = None,
    filter_file: Optional[str] = None,
    output_file: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:

    if isinstance(config, str):
        config = OmegaConf.load(config)

    data_directory = config.data_directory.format(__dir__=__DIR__)
    mordred_config = config.mordred_features
    remove_cidm = config.networks.remove_cidm
    chemical_as_int = config.networks.chemical_as_int

    if input_file is None:
        input_file = path.join(data_directory,
                               mordred_config.input_file)

    if filter_file is None:
        filter_file = mordred_config.filter_file

    if output_file is None:
        output_file = path.join(data_directory,
                                mordred_config.output_file)

    print("Loading drugs file...")
    drugs_df = load_drugs_file(input_file,
                               remove_cidm=remove_cidm,
                               chemical_as_int=chemical_as_int).set_index("chemical")

    if drugs is not None:
        drugs_df = drugs_df[drugs_df.index.isin(drugs)]

    if limit is not None:
        drugs_df = drugs_df.iloc[:limit]

    pbar = tqdm(zip(drugs_df.index, drugs_df["SMILES_string"]), total=len(drugs_df))
    pbar.set_description("Extracting Mordred features")
    features = None

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    for c, smile in pbar:
        m = Chem.MolFromSmiles(smile)
        if m is None:  continue

        result = Calculator(descriptors.all, ignore_3D=True)(m).asdict()
        for k, v in result.items():
            if is_missing(v):
                result[k] = None

        if features is None:
            features = dict(chemical=[c])
            features.update({k: [v] for k, v in result.items()})
        else:
            features["chemical"].append(c)
            for k, v in result.items():
                features[k].append(v)
    warnings.filterwarnings("default", category=RuntimeWarning)

    features = pd.DataFrame(features).set_index("chemical")

    if filter_file is not None:
        with open(path.abspath(path.join(data_directory, filter_file)), "r") as f:
            keep_features = f.read().splitlines()

        features = features[keep_features]

    features.to_csv(output_file, sep=get_delim(output_file))
    print("\nDone!")

    return features
