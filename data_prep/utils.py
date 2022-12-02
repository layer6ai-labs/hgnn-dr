import pandas as pd
from os import path

__DIR__ = path.dirname(path.realpath(__file__))


def convert_chemicals(c, col="chemical", to=int):
    if isinstance(c, str):
        if to==int:  return int(c[4:].lstrip("0"))
        else:        raise TypeError("str can only be converted to int")

    if isinstance(c, int):
        if to==str:  return f"CIDs{c:08d}"
        else:        raise TypeError("int can only be converted to str")

    elif isinstance(c, pd.Series):
        return c.apply(convert_chemicals, to=to)

    elif isinstance(c, pd.DataFrame):
        assert col in c.columns, "Must supply a valid column name."
        c[col] = c[col].apply(convert_chemicals, to=to)
        return c


def get_delim(file):
    ext = path.splitext(file.strip(".gz"))[-1]
    return {
        ".csv": ",",
        ".tsv": "\t",
        ".txt": " ",
    }[ext]


def load_drugs_file(drugs_file: str, chemical_as_int=False, remove_cidm=False):
    drugs_df = pd.read_csv(drugs_file, sep=get_delim(drugs_file))

    if remove_cidm:
        is_cidm = lambda d: d.startswith("CIDm")
        drugs_df = drugs_df[~drugs_df["chemical"].apply(is_cidm)]

    if chemical_as_int:
        drugs_df = convert_chemicals(drugs_df, col="chemical", to=int)

    return drugs_df
