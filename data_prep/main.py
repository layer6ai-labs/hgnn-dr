import argparse
from omegaconf import DictConfig, OmegaConf
from os import path
from typing import Optional, Sequence, Union

from .drugs import create_drugs_file
from .networks import ProteinDrugNetwork
from .source_files import SourceFiles
from .unirep_features import UniRepFeatureExtractor
from .rdkit_features import create_rdkit_features_file
from .mordred_features import create_mordred_features_file

__DIR__ = path.dirname(path.realpath(__file__))

import sys
sys.path.append(path.join(__DIR__, "..", "baselines"))
from build_features import build_processed_features

PIPELINE = [
    "downloads",
    "drugs",
    "networks",
    "unirep",
    "rdkit",
    "mordred",
    "process_features"
]


def main(
    config: Union[DictConfig, str] = path.join(__DIR__, "config.yaml"),
    force_download: bool = False,
    skip: Optional[Sequence[str]] = []
) -> None:

    print("\nPerforming data prep")
    if isinstance(config, str):
        config = OmegaConf.load(config)
    print()

    if "downloads" in skip:
        print("Skipping downloads")
    else:
        # Download all source files (if needed)
        SourceFiles(config).download_all(force=force_download)
    print()

    if "drugs" in skip:
        print("Skipping creation of drugs file")
    else:
        # Filter chemical aliases for drugs only
        print("Creating drugs file")
        create_drugs_file(config=config)
    print()

    # Create ProteinDrugNetwork for each organism and cache to json
    #   or load from cache and grab drugs / proteins
    drugs = set()
    proteins = dict()
    for organism in config.organisms:
        if ("networks" in skip) or (f"networks.{organism}" in skip):
            print(f"Loading ProteinDrugNetwork for {organism}")
            network = ProteinDrugNetwork(organism=organism, config=config).from_json()
        else:
            print(f"Creating ProteinDrugNetwork for {organism}")
            network = ProteinDrugNetwork(organism=organism, config=config).build()
            network.to_json()

        proteins[organism] = network.proteins
        drugs |= network.drugs
    print()

    if "unirep" in skip:
        print("Skipping Unirep protein feature extraction")
    else:
        print("Creating UniRep protein features (this may take a while)")
        unirep = UniRepFeatureExtractor(config=config)
        for organism, org_proteins in proteins.items():
            # create unirep features for organism
            if f"unirep.{organism}" in skip:
                print(f"Skipping for organism {organism}")
            else:
                unirep.extract_features(organism=organism, proteins=org_proteins)
    print()

    # Create drug features
    if "rdkit" in skip:
        print("Skipping RDKit drug feature creation")
    else:
        print("Creating RDKit drug features")
        create_rdkit_features_file(drugs=drugs, config=config)
    print()

    if "mordred" in skip:
        print("Skipping Mordred drug feature creation")
    else:
        print("Creating Mordred drug features (this may take a while)")
        create_mordred_features_file(drugs=drugs, config=config)
    print()

    if "process_features" in skip:
        print("Skipping feature processing")
    else:
        print("Processing features")
        data_directory = config.data_directory.format(__dir__=__DIR__)
        build_processed_features(
            unirep_paths = [path.join(data_directory, f"{org}.unirep_features.npz")
                            for org in config.organisms],
            rdkit_path = path.join(data_directory, "rdkit_features.tsv.gz"),
            mordred_path = path.join(data_directory, "mordred_features.tsv.gz"),
        )
    print()

    print("Finished data prep!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for data prep.")
    parser.add_argument("--config",
                        type=str,
                        default=path.join(__DIR__, "config.yaml"),
                        help="config for the data prep module")
    parser.add_argument("--force-download",
                        action="store_true",
                        help="redownload all files, even if present")
    parser.add_argument("--skip",
                        nargs="*",
                        default=[],
                        help=f"list of flags to skip pipeline steps; a subset of {PIPELINE}. "
                              "For 'networks' and 'unirep', you can also skip specific "
                              "organisms using dot notation, e.g. 'unirep.5664'")
    args = parser.parse_args()
    main(**vars(args))
