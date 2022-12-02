import argparse
from os import path

from gcn.utils.train_utils import add_flags_from_config

__DIR__ = path.dirname(path.realpath(__file__))
_DATA_DIR = path.abspath(path.join(__DIR__, "..", "data"))

config_args = {
    'training_config': {
        'log': ('inital_run', 'None for no logging'),
        'do-kfold': (False, 'whether or not to do k-fold CV; # of folds depends on fold-prop'),
        'lr': (1e-2, 'learning rate'),  # 1e-2 for default, 1e-3 for resnet
        'dropout': (0.01, 'dropout probability'),
        'cuda': (0, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (200, 'maximum number of epochs to train for'),
        'weight-decay': (0, 'l2 regularization strength'),
        'momentum': (0.8, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1236, 'seed for training'),
        'eval-freq': (10, 'how often to compute val metrics (in epochs)'),
        'save': (1, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to out_path/dataset/date'),
        'save-scores':([], 'splits to save best model scores for p-c pairs; subset of ["train", "test", "inference"]. Can be []'),
        'save-scores-batchsize': (100_000, 'batch size of edges to process for score saving'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs')
    },
    'model_config': {
        'task': ('lp', 'which tasks to train on, can be any of [lp, bi_lp]'),
        'model': ('HGCN', 'which encoder to use, can be any of [ MLP, GCN, HGCN]'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, PoincareBall]'),
        'dim': (25, 'embedding dimension'),
        'scale': (0.1, 'scale for embedding init'),
        'num-layers': (1, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('tanh', 'which activation function to use (or None for no activation)'),
        'double-precision': ('0', 'whether to use double precision'),
        'tangent-agg-layers':(2, 'number of layers of tangent aggregation'),
        'tangent-agg':('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN')
    },
    'data_config': {
        'data_prep_path': (f'{_DATA_DIR}/data_prep','path to data prep files'),
        'flagged_chems_path': (f"{_DATA_DIR}/filters/chems_flagged_for_removal.tsv",
                               'path of chems to be removed, or None'),
        'features_path': (f'{_DATA_DIR}/baselines/features.pkl', 'path to features file'),
        'out_path': (f"{_DATA_DIR}/gcn_output", 'path to store output log and model files'),
        'dataset': ('test_run', 'name of the dataset'),
        'skip-feats': (0, 'whether to skip loading features (for baselines)'),
        'target_organism': (5664, 'organism to predict drugs for'),
        'other_organisms': ([5833, 5691], 'organisms used to augment dataset, can include [5833, 5691]'),
        'include_other_organisms':([5833,5691], 'additional organism interactions to use in training, None for only target'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'fold-prop': (0.1, 'proportion of edges in each fold for link prediction'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'max-chem-degree':(500, 'exclude chem from test and valid with degree greater than threshold.'),
        'split-seed': (4569, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
