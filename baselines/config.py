from omegaconf import OmegaConf
from os import path

__DIR__ = path.dirname(path.realpath(__file__))

BASE_CONF = OmegaConf.load(f"{__DIR__}/config.yaml")
DATASET_CONF = BASE_CONF.dataset
MODELS_CONF = BASE_CONF.models

DATA_DIR = path.abspath(BASE_CONF.data_directory.format(__dir__=__DIR__))
OUT_DIR = path.abspath(f"{DATA_DIR}/{DATASET_CONF.directory}")

RANDOM_STATE = BASE_CONF.random_state
MODEL_CONSTANTS = dict(MODELS_CONF.constants)

def OC_CONVERT(config):
    try:
        return OmegaConf.to_object(config)
    except ValueError:
        return config
