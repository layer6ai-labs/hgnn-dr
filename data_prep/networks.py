from __future__ import annotations

import json
import pandas as pd
from omegaconf import OmegaConf
from os import path
from typing import Optional, Union, Sequence

from .utils import convert_chemicals, get_delim, load_drugs_file

__DIR__ = path.dirname(path.realpath(__file__))


class ProteinDrugNetwork:
    def __init__(self,
        config: str = path.join(__DIR__, "config.yaml"),
        data_directory: Optional[str] = None,
        source_file_directory: Optional[str] = None,
        organism=5664,
    ):

        if isinstance(config, str):  config = OmegaConf.load(config)

        self.data_directory = (config.data_directory.format(__dir__=__DIR__)
                               if data_directory is None else data_directory)

        self.source_file_directory = (
            path.join(self.data_directory, config.source_files.subdirectory)
            if source_file_directory is None else source_file_directory
        )

        self.organism = organism
        self.net_config = config.networks


    def build(self) -> ProteinDrugNetwork:
        self.protein_links = self.extract_links("protein_links", node_columns_equal=True)
        self.pc_links = self.extract_links("protein_chemical_links", node_columns_equal=False)

        test_drugs = self.net_config.test_drugs
        if test_drugs is not None:
            self.test_drug_pc_links = self.extract_links("protein_chemical_links",
                                                         node_columns_equal=False,
                                                         scores_to_use="all")


            is_test_drug = self.test_drug_pc_links["chemical"].isin(test_drugs)
            self.test_drug_pc_links = self.test_drug_pc_links[is_test_drug]

            self.pc_links = self.pc_links[~self.pc_links["chemical"].isin(test_drugs)]
        else:
            self.test_drug_pc_links = pd.DataFrame(columns=self.pc_links.columns)

        # Filter PC links by drugs
        drugs = load_drugs_file(
            path.join(self.data_directory, self.net_config.drugs_file)
        )["chemical"].unique()

        self.pc_links = self.pc_links[self.pc_links["chemical"].isin(drugs)]

        # Filter PC links and test PC links by actions
        pc_actions = self.extract_pc_actions()

        actions_set = {(p,c) for p,c in zip(pc_actions["protein"], pc_actions["chemical"])}
        self.pc_links = keep_links_in_set(self.pc_links, actions_set)
        self.test_drug_pc_links = keep_links_in_set(self.test_drug_pc_links, actions_set)

        # convert chemicals as ints
        if self.net_config.chemical_as_int:
            self.pc_links = convert_chemicals(self.pc_links)
            self.test_drug_pc_links = convert_chemicals(self.test_drug_pc_links)

        # drop duplicate links and reset index
        self.pc_links = self.pc_links.drop_duplicates().reset_index(drop=True)
        self.test_drug_pc_links = self.test_drug_pc_links.drop_duplicates().reset_index(drop=True)

        self.proteins = set(self.pc_links.protein) | set(self.test_drug_pc_links.protein)
        self.chemicals = set(self.pc_links.chemical) | set(self.test_drug_pc_links.chemical)
        self.drugs = self.chemicals     # just a reference
        return self


    def extract_links(self,
        source: str,
        node_columns_equal: bool = False,
        scores_to_use: Optional[Union[Sequence[str], str]] = None,
    ) -> pd.DataFrame:

        params = self.net_config[source]
        source = self._get_filepath(params.source, source=True)
        node_columns = params.node_columns

        df = pd.read_csv(source, sep=get_delim(source))

        # Remove CIDms
        if self.net_config.remove_cidm:
            is_cidm = df[node_columns].applymap(lambda x: x.startswith("CIDm"))
            df = df[~is_cidm.any(axis=1)]

        if scores_to_use is None:
            scores_to_use = params.scores_to_use

        # Keep only rows with the scores_to_use
        if scores_to_use == "all":
            df = df[node_columns]
        else:
            rows_with_scores = (df[scores_to_use]>0).any(axis=1)
            df = df.loc[rows_with_scores, node_columns]

        # Sort proteins in links, e.g. so duplicate links can be removed: (x,y) vs (y,x)
        if node_columns_equal:
            df_values = df.values
            df_values.sort(axis=1)
            df = pd.DataFrame(df, columns=df.columns, index=df.index)

        return df.drop_duplicates(node_columns).reset_index(drop=True)


    def extract_pc_actions(self) -> pd.DataFrame:
        params = self.net_config.protein_chemical_actions
        source = self._get_filepath(params.source, source=True)
        node_columns = params.node_columns

        df = pd.read_csv(source, sep=get_delim(source)).fillna("n/a")

        # Remove CIDms
        if self.net_config.remove_cidm:
            is_cidm = df[node_columns].applymap(lambda x: x.startswith("CIDm"))
            df = df[~is_cidm.any(axis=1)]

        # note columns that can't be kept consistently with repeats
        invalid_cols = ["item_id_a", "item_id_b", "a_is_acting"]
        data_cols = [col for col in df.columns if col not in invalid_cols]

        # Sort item_id_a vs. item_id_b: outcome should be protein < chemical
        pc_cols = df[["item_id_a", "item_id_b"]].values.copy()
        pc_cols.sort(axis=1)
        pc_cols = pd.DataFrame(pc_cols, columns=["protein", "chemical"], index=df.index)
        actions = pd.concat([pc_cols, df[data_cols]], axis=1)

        # If a mode is to be removed is present for some (protein, chemical) pair,
        # remove all rows with the same protein / chemical.
        remove_modes = params.remove_modes
        if remove_modes is not None:
            pcx = zip(actions["protein"], actions["chemical"], actions["mode"])
            pc_to_remove = {(p,c) for (p,c,x) in pcx if x in remove_modes}
            # Finish removal after dropping duplicates

        # Drop rows with the same PC pairs, keep row w/ highest score
        actions.sort_values("score", ascending=False, inplace=True)
        actions.drop_duplicates(["protein", "chemical"], inplace=True)

        # Finish removing modes after reducing # of rows
        if remove_modes is not None:
            pcx = zip(actions["protein"], actions["chemical"], actions.index)
            rows_to_remove = [x for (p,c,x) in pcx if (p,c) in pc_to_remove]
            actions.drop(rows_to_remove, inplace=True)

        return actions.reset_index(drop=True)


    def to_json(self) -> None:
        network_dict = dict(
            pc_links = self.pc_links.to_dict("list"),
            test_drug_pc_links = self.test_drug_pc_links.to_dict("list")
        )

        with open(self._get_filepath(self.net_config.output_file), "w") as f:
            json.dump(network_dict, f)


    def from_json(self) -> ProteinDrugNetwork:
        with open(self._get_filepath(self.net_config.output_file), "r") as f:
            network_dict = json.load(f)

        self.pc_links = pd.DataFrame(network_dict["pc_links"])
        self.test_drug_pc_links = pd.DataFrame(network_dict["test_drug_pc_links"])

        self.proteins = set(self.pc_links.protein) | set(self.test_drug_pc_links.protein)
        self.chemicals = set(self.pc_links.chemical) | set(self.test_drug_pc_links.chemical)
        self.drugs = self.chemicals     # just a reference
        return self


    def _get_filepath(self, file, source=False):
        dir = self.source_file_directory if source else self.data_directory
        return path.join(dir, file.format(org=self.organism))


def keep_links_in_set(links, keep_set):
    links_zip = zip(links["protein"], links["chemical"], links.index)
    rows_to_keep = (i for (p,c,i) in links_zip if (p,c) in keep_set)
    return links.loc[rows_to_keep, :]
