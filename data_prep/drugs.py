import pyarrow as pa
import pyarrow.compute as pc
from omegaconf import OmegaConf, DictConfig
from os import path
from pyarrow import csv
from typing import Callable, Optional, Union, Sequence

from .utils import get_delim

__DIR__ = path.dirname(path.realpath(__file__))


def create_drugs_file(
    config: Union[DictConfig, str] = path.join(__DIR__, "config.yaml"),
    chemical_aliases_source: Optional[str] = None,
    chemicals_source: Optional[str] = None,
    drug_alias_filter: Optional[Callable[[pa.Table], pa.Table]] = None,
    output_file: Optional[str] = None,
) -> None:
    """Filter chemical aliases to only keep rows corresponding to drug ids.

    Keyword arguments:
    input_file -- path to chemical aliases .tsv(.gz) file, e.g. chemicals.v5.0.tsv.gz
    output_file -- path to write filtered rows
    filt -- operates on a Table and returns a filtered Table

    config -- optional arguments will defaults to this config
    """

    if isinstance(config, str):
        config = OmegaConf.load(config)

    data_directory = config.data_directory.format(__dir__=__DIR__)
    drugs_config = config.drugs

    if chemical_aliases_source is None:
        chemical_aliases_source = path.join(data_directory,
                                            config.source_files.subdirectory,
                                            drugs_config.chemical_aliases_source)

    if chemicals_source is None:
        chemicals_source = path.join(data_directory,
                                     config.source_files.subdirectory,
                                     drugs_config.chemicals_source)

    if drug_alias_filter is None:
        drug_alias_filter = OmegaConf.to_object(drugs_config.alias_regexp_and_filter)

    if output_file is None:
        output_file = path.join(data_directory, drugs_config.output_file)

    print("Loading drug aliases")
    drug_aliases = load_drug_aliases(chemical_aliases_source, drug_alias_filter)

    print("Loading drug properties")
    drugs = load_drugs(chemicals_source, drug_aliases["chemical"].to_numpy())

    drugs = drug_aliases.join(drugs, "chemical")

    print("Writing drugs file")
    csv.write_csv(
        drugs,
        output_file,
        write_options=csv.WriteOptions(delimiter=get_delim(output_file))
    )


def regexp_filter(**kwargs) -> Callable[[pa.Table], pa.Table]:
    """Constructs a drug filt matching the intersection of column-regex pairs."""

    if len(kwargs) > 0:
        def filt(table):
            filt = None

            for (col, r) in kwargs.items():
                col_filt = pc.match_substring_regex(table[col], r)
                filt = col_filt if filt is None else pc.and_(filt, col_filt)

            return table.filter(filt)

    else:
        filt = lambda table: table

    return filt


def load_drug_aliases(
    input_file: str,
    filt: Union[Callable[[pa.Table], pa.Table], "dict[str, str]"]
) -> pa.Table:
    """Filter chemical aliases to only keep rows corresponding to drug ids.

    Keyword arguments:
    input_file -- path to chemical aliases .tsv(.gz) file, e.g. chemicals.v5.0.tsv.gz
    output_file -- path to write filtered rows
    filt -- operates on a Table and returns a filtered Table, or a dict of
            column to regexp mappings to create an AND filter matching the expresions

    config -- optional arguments will defaults to this config
    """

    if isinstance(filt, dict):
        filt = regexp_filter(**filt)

    aliases = csv.read_csv(
        input_file,
        parse_options=csv.ParseOptions(delimiter=get_delim(input_file))
    )

    aliases = filt(aliases)

    aliases = pa.concat_tables([
        aliases.select(["stereo_chemical", "alias", "source"])
                    .rename_columns(["chemical", "alias", "source"]),
        aliases.select(["flat_chemical", "alias", "source"])
                    .rename_columns(["chemical", "alias", "source"]),
    ])

    return aliases


def load_drugs(
    input_file: str,
    drug_chemicals: Sequence,
):
    chemicals = csv.read_csv(
        input_file,
        parse_options=csv.ParseOptions(delimiter=get_delim(input_file))
    )

    expr = pc.field("chemical").isin(drug_chemicals)
    drugs = chemicals.filter(expr)

    return drugs
