import pandas as pd
from omegaconf import DictConfig, OmegaConf
from os import path
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from tqdm import tqdm
from typing import Optional, Union, Sequence

from .utils import get_delim, load_drugs_file


__DIR__ = path.dirname(path.realpath(__file__))


def create_rdkit_features_file(
    drugs: Optional[Sequence[Union[str, int]]] = None,
    config: Union[DictConfig, str] = path.join(__DIR__, "config.yaml"),
    input_file: Optional[str] = None,
    filter_file: Optional[str] = None,
    output_file: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:

    RDLogger.DisableLog('rdApp.*')

    if isinstance(config, str):
        config = OmegaConf.load(config)

    data_directory = config.data_directory.format(__dir__=__DIR__)
    rdkit_config = config.rdkit_features
    remove_cidm = config.networks.remove_cidm
    chemical_as_int = config.networks.chemical_as_int

    if input_file is None:
        input_file = path.join(data_directory,
                               rdkit_config.input_file)

    if filter_file is None:
        filter_file = rdkit_config.filter_file

    if output_file is None:
        output_file = path.join(data_directory,
                                rdkit_config.output_file)

    print("Loading drugs file...")
    drugs_df = load_drugs_file(input_file,
                               remove_cidm=remove_cidm,
                               chemical_as_int=chemical_as_int).set_index("chemical")

    if drugs is not None:
        drugs_df = drugs_df[drugs_df.index.isin(drugs)]

    if limit is not None:
        drugs_df = drugs_df.iloc[:limit]

    def get_features(drugs_df):
        df = dict(chemical=[])
        df.update({desc: [] for desc, _ in Descriptors.descList})

        pbar = tqdm(zip(drugs_df.index, drugs_df["SMILES_string"]), total=len(drugs_df))
        pbar.set_description("Extracting RDKit features")
        for c, smile in pbar:
            m = Chem.MolFromSmiles(smile)
            if m is None:  continue

            df["chemical"].append(c)
            for desc, descfun in Descriptors.descList:
                df[desc].append(descfun(m))

        return pd.DataFrame(df).set_index("chemical")

    features = get_features(drugs_df)

    if filter_file is not None:
        with open(path.join(data_directory, filter_file), "r") as f:
            keep_features = f.read().splitlines()

        features = features[keep_features]

    features.to_csv(output_file, sep=get_delim(output_file))
    print("\nDone!")

    return features
