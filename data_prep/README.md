# Data Prep

## Running the pipeline

Data prep is performed on Python 3.8. To run the data prep pipeline, run from the project (i.e. parent) directory

```bash
python -m data_prep.main
```

By default, all data prep files are stored in `data/data_prep`. Source files are downloaded into `data/data_prep/source_files` (relative to the project directory).

### Optional arguments

* `--config CONFIG`: a YAML config file for data prep. See the [Config](#config) section.

* `--force-download`: this flag will force downloads for source files even if they are already present.

* `--skip [SKIP [SKIP ...]]`: a list of pipeline steps to skip.

  * The available pipeline steps are `downloads`, `drugs`, `networks`, `unirep`, `rdkit`, `mordred` and `process_features`. See the [Pipeline](#pipeline) section.

  * For the `networks` and `unirep` pipelines, it is possible to choose to skip the pipeline for specific organisms via taxid using dot notation, e.g. `networks.TAXID`.

  * For instance, skip downloads and network creation for *P. Falciparum* (taxid `5833`) by running

    ```bash
    python -m data_prep.main --skip downloads networks.5833
    ```

### Running time

Running the pipeline can take several hours. The main bottlenecks occur in

* `downloads`: More specifically, downloading the `chemicals` and `chemical_aliases` files;

* `unirep`: Creating UniRep features;

* `mordred`: Creating Mordred features, albeit to a lesser extent.


&nbsp;

## Config

The config file contains settings for all pipeline stages. This includes

* The `data_directory` in which data prep files are stored;
* Locations and local names of `source_files`, as well as the source file subdirectory;
* Settings for extracting `drugs` information;
* Settings for constructing protein-drug `networks`;
* Settings for constructing `unirep` protein features, as well as `rdkit` and `mordred` chemical descriptor features.

The default config file is `data_prep/config.yaml`


&nbsp;

## Pipeline

**Jump to:**
  [`downloads`](#downloads-download-source-files)
| [`drugs`](#drugs-compile-drugs-file)
| [`networks`](#networks-create-proteindrugnetwork-for-organism)
| [`unirep`](#unirep-create-unirep-protein-features)
| [`rdkit`](#rdkit-create-rdkit-chemical-features)
| [`mordred`](#mordred-create-mordred-chemical-features)

&nbsp;

### `downloads`: download source files

The config for this step is provided under `source_files`. By default, downloaded files are stored in the `../data/data_prep/source_files` subdirectory.

Other than the `subdirectory`, remaining entries specify a local `file` name and the URL to retrieve the file from. In `data_prep.main`, the `SourceFiles().download_all()` method is invoked to download each of these files.

Source files come from the [STITCH](http://stitch.embl.de/) and [STRING](https://string-db.org/) databases.

&nbsp;

### `drugs`: compile drugs file

The `create_drugs_file()` function in `data_prep.drugs` extracts drug information by doing the following:

1. Drugs are identified from `chemical_aliases` source file, by filtering rows to keep chemicals where both of the following properties are present:

    * An `alias` starting with `D*` or `DB*`, followed only by numerical digits;

    * A `source` containing the substring `KEGG` or `DrugBank`.

  This filtering step is specified in the config under the `drugs.alias_regexp_and_filter` entry.

2. Join drug aliases with the `chemicals` source file to associate each drug with a SMILES string and name.

&nbsp;

### `networks`: create `ProteinDrugNetwork` for organism

For a given organism, its `ProteinDrugNetwork` is created using the corresponding `protein_links` and `protein_chemical_links` source files based on a number of filtering steps:

1. Links are filtered so that only those based on experimental evidence are kept, as specified in `config.yaml`.

2. Only links involving chemicals identified as drugs are kept; see `drugs`.

3. Any links indicated to involve a catalysis action is removed, based on the `protein_chemical_actions` source file.

In addition, the following post-processing steps are performed:

1. Links involving known test drugs miltefosine and pentamidine are removed from the existing network if present and are stored as a complementary network, *regardless of whether the links come from experimental evidence*.

2. We only keep stereo chemicals, i.e. remove any chemicals with id starting with `CIDm*`. Furthermore, we strip the prefix from all chemicals and keep the id as an integer.

3. The resulting network is saved as a JSON file for easy reloading.

&nbsp;

### `unirep`: create UniRep protein features

UniRep is a LSTM model that extracts features from protein sequences; see the [original publication](https://www.biorxiv.org/content/10.1101/589333v1). We use the [`jax-unirep`](https://github.com/ElArkk/jax-unirep) implementation of the model.

FASTA sequences for each protein in each organism's `ProteinDrugNetwork` are obtained using the corresponding `protein_sequences` source file, and fed into the UniRep model to obtain a feature representation for that protein.

This takes a very long time and has high memory usage, so the representations can only be extracted in batches.

&nbsp;

### `rdkit`: create RDKit chemical features

We use [RDKit](https://www.rdkit.org/) to calculate molecular descriptors for each drug in over the union of ProteinDrugNetworks for all organisms. Descriptors are calculated over a SMILES string representation of each drug.

We manually reviewed the profiles of these descriptors and removed constant, sparse, and highly skewed features. The features that are to be kept are specified in `rdkit_features.yaml`.

&nbsp;

### `mordred`: create Mordred chemical features

The process here is similar to the `rdkit` step, except the [`mordred`](http://mordred-descriptor.github.io/documentation/v0.1.0/index.html) chemical descriptor calculator is used (see [original publication](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0258-y)).

Features to be kept after manual review are specified in `mordred_features.yaml`.

The `mordred` library computes a significantly larger number of molecular descriptors for each SMILES string than `rdkit`, and as a result can take a somewhat long time.

&nbsp;

### `process_features`: perform median imputation and PCA on features

This process is simply called from `baselines/build_features.py`. The config is almost entirely handled by `baselines`.

The `main` script simply includes this as part of the data prep pipeline and ensures that the input files are correctly specified based on the outputs of the earlier stages.
