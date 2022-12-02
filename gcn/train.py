from __future__ import division
from __future__ import print_function

import warnings     # ignore Pandas FutureWarning
warnings.simplefilter(action='ignore', category=FutureWarning)

import json
import logging
import os
import pickle
import time
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import ast


if __name__ == '__main__':
    import sys
    script_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(f"{script_dir}/../")

    # This allows gcn.train module to be imported and not run as __main__
    import gcn.train as tr
    from gcn.config import parser
    from gcn.utils.data_utils import Data
    from gcn.utils.train_utils import get_dir_name

    args = parser.parse_args()
    if args.save:
        if not args.save_dir:
            dt = datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(args.out_path, args.dataset, date)
            args.save_dir = get_dir_name(models_dir)

        logging.basicConfig(
            level=logging.INFO,
            handlers=[logging.FileHandler(os.path.join(args.save_dir, 'log.txt')),
                      logging.StreamHandler()]
        )
        logging.info('Saving model output to {}'.format(args.save_dir))
        for arg in vars(args):
            logging.info(arg + '=' + str(getattr(args, arg)))

    # === prepare data
    data = Data(args=args).build()

    tr.train(data, args)
    sys.exit()


from .models.base_models import LPModel
from .utils.train_utils import format_metrics


def train(data, args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    logging.getLogger().setLevel(logging.INFO)
    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    #------ Set Network Params
    #args.n_nodes, args.feat_dim = data['features'].shape
    args.num_proteins = data.num_proteins
    args.num_chemicals = data.num_chemicals
    args.n_nodes = data.num_proteins+data.num_chemicals
    args.c = 1.0
    args.r = 2.0
    args.t = 1.0


    #------ Set Training Params
    args.patience = args.epochs if not args.patience else int(args.patience)

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Train folds
    if args.do_kfold:
        num_folds = len(data.data["fold_edges"])

        logging.info(f"Training model over {num_folds} folds")
        fold_models, fold_summaries = [], []
        for fold in range(num_folds):
            model, summary = train_split(data, args, fold=fold)
            fold_models.append(model)
            fold_summaries.append(summary)

        # Get average metrics across folds
        if args.save_dir is not None:
            stats = stats_by_epoch(fold_summaries)

            # Write means to tensorboard
            for split, metrics in stats.items():
                mean_metrics = metrics.set_index("epoch")
                mean_metrics = mean_metrics[filter(lambda col: col.endswith("_mean"),
                                                   mean_metrics.columns)]

        # Report average best validation / test scores
        best_perfs = {s: pd.DataFrame() for s in fold_summaries[0]
                      if (s!="best_epoch" and s.startswith("best_"))}
        for summary in fold_summaries:
            for s in best_perfs:
                best_perfs[s] = best_perfs[s].append(summary[s], ignore_index=True)

        best_means = {s[5:]: df.mean().to_dict() for s, df in best_perfs.items()}
        best_stds = {s[5:]: df.std().to_dict() for s, df in best_perfs.items()}

        logging.info("\n".join([
            f"Finished. Average best perf. across {num_folds} folds:",
            "  "+format_metric_stats(best_means["val"], best_stds["val"], 'val'),
            "  "+format_metric_stats(best_means["test"], best_stds["test"], 'test')]))

        best_val_roc = lambda s: s["best_val"]["roc"]
        best_fold = np.argmax([best_val_roc(s) for s in fold_summaries])
        best_model, best_summary = fold_models[best_fold], fold_summaries[best_fold]

        logging.info(f"Best epoch (fold {best_fold}): {best_summary['best_epoch']}")
        logging.info(f"Best model val performance: {format_metrics(best_summary['best_val'])}")
    else:
        logging.info("Training model on all data")
        best_model, best_summary = train_split(data, args, fold=None)

    logging.info(f"Best model test performance: {format_metrics(best_summary['best_test'])}")

    best_model.eval()
    if args.save:
        best_emb = best_model.encode(data.features, data.adj_train_norm)
        save_best_model(args, best_model, best_emb)

        # Get scores from best model
        for split in args.save_scores:
            logging.info(f"Saving {split} scores from best model")
            scores = compile_pc_scores(data, args, best_model, split, args.save_scores_batchsize)
            print(scores, end="\n\n")
            scores.to_csv(f"{args.save_dir}/{split}_scores.csv", index=False)


def train_split(data, args, fold=None):
    do_val = fold is not None
    subdir = fold if do_val else "all"

    data.set_adj_train(args, fold=fold)
    save_dir = None if args.save_dir is None else args.save_dir + f"/{subdir}"

    args.nb_false_edges = data.num_train_edges_false
    args.nb_edges = data.num_train_edges
    args.feat_dim = data.features.shape[1]

    # Model and optimizer
    Model, Optimizer, LRScheduler = get_train_objs(args)
    model = Model(args)
    logging.info(str(model))

    optimizer = Optimizer(params=model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)

    lr_scheduler = LRScheduler(optimizer, step_size=int(args.lr_reduce_freq),
                               gamma=float(args.gamma))

    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 and torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        if args.normalize_adj:
            data.adj_train_norm.to(args.device)

    # Train model
    summary = { "best_val": model.init_metric_dict(), "best_epoch": 0,
                "train": pd.DataFrame(), "val": pd.DataFrame() }
    epochs_reported = []
    best_emb = None

    t_total = time.time()
    stop_counter = 0
    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()

        emb = model.encode(data.features, data.adj_train_norm)
        train_loss, *_ = model.compute_loss_scores(emb, data, 'train')
        train_loss.backward()

        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            all_params = list(model.parameters())
            for param in all_params:
                torch.nn.utils.clip_grad_norm_(param, max_norm)
        optimizer.step()
        lr_scheduler.step()

        # Log train loss
        if torch.__version__ > "1.4.0":
            lr_val = lr_scheduler.get_last_lr()[0]
        else:
            lr_val = lr_scheduler.get_lr()[0]

        epoch_str = 'Epoch: {:04d}'.format(epoch + 1)
        logging.info(" ".join([epoch_str,
                               'lr: {}'.format(lr_val),
                               'loss: {:.4f}'.format(train_loss),
                               'time: {:.4f}s'.format(time.time() - t)]))

        if (epoch + 1) % args.eval_freq == 0:
            model.eval()
            with torch.no_grad():
                emb = model.encode(data.features, data.adj_train_norm)
                train_metrics = model.compute_metrics(emb, data, 'train')
                summary["train"] = summary["train"].append(train_metrics, ignore_index=True)
                logging.info(f"{epoch_str} {format_metrics(train_metrics, 'train')}")

                if do_val:
                    val_metrics = model.compute_metrics(emb, data, 'val', False)
                    summary["val"] = summary["val"].append(val_metrics, ignore_index=True)
                    logging.info(" ".join([epoch_str,
                                           format_metrics(val_metrics, 'val')]))

                epochs_reported.append(epoch)

                if do_val:
                    if model.has_improved(summary["best_val"], val_metrics):
                        stop_counter = 0
                        summary["best_val"] = val_metrics
                        summary["best_epoch"] = epoch
                        best_emb = emb.cpu()
                        if args.save:
                            np.save(os.path.join(save_dir, 'embeddings.npy'),
                                    best_emb.detach().numpy())
                    else:
                        stop_counter += 1
                        if stop_counter == args.patience and epoch > args.min_epochs:
                            logging.info("Early stopping")
                            break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    model.eval()

    with torch.no_grad():
        best_emb = model.encode(data.features, data.adj_train_norm)
        summary["best_test"] = model.compute_metrics(best_emb, data, 'test',False)

    if do_val:
        logging.info(f"Val set results: {format_metrics(summary['best_val'], 'val')}")
    logging.info(f"Test set results: {format_metrics(summary['best_test'], 'test')}")

    summary["train"].insert(0, "epoch", epochs_reported)

    if do_val:
        summary["val"].insert(0, "epoch", epochs_reported)

    return model, summary

def stats_by_epoch(summaries, splits=["train", "val"]):
    "Takes summaries and returns the statistics of their metrics by epoch"

    def merge_summary(left, right, suffixes):
        def lr(split):
            rn = lambda suffix: lambda col: f"{col}{suffix}" if col!="epoch" else col
            return (left[split].rename(rn(suffixes[0]), axis=1),
                    right[split].rename(rn(suffixes[1]), axis=1))
        return {split: pd.merge(*lr(split), on="epoch") for split in splits}

    merged_summary = summaries[0]
    for fold, summary in enumerate(summaries[1:], 1):
        suffixes = ("_0" if fold==1 else "", f"_{fold}")
        merged_summary = merge_summary(merged_summary, summary, suffixes)

    stats = dict()
    for split in splits:
        metrics = merged_summary[split]
        metric_stats = metrics[["epoch"]].copy()   # new df with epochs

        metric_names = [
            col for col in summaries[0][split].columns if col!="epoch"]
        for name in metric_names:
            cols = [col for col in metrics.columns if col.startswith(name)]
            metric_stats[f"{name}_mean"] = metrics[cols].mean(axis=1)
            metric_stats[f"{name}_std"] = metrics[cols].std(axis=1)

        stats[split] = metric_stats
    return stats

def compile_pc_scores(data, args, model, split, batch_size=100_000):
    """Compile model scores for protein-chemical pairs"""
    model.eval()
    with torch.no_grad():
        emb = model.encode(data.features, data.adj_train_norm)
        edges = [model.get_edges_false(data, split, sample=False), data.data[f'{split}_edges']]
        truth = np.concatenate([np.zeros(len(edges[0]), dtype=bool),
                                np.ones(len(edges[1]), dtype=bool)])
        edges = torch.cat(edges).numpy()

        bat_intv = torch.cat([torch.arange(0, len(edges), batch_size),
                              torch.tensor([len(edges)])])
        if bat_intv[-1] == bat_intv[-2]:
            bat_intv = bat_intv[:-1]

        scores = []
        for i in tqdm(range(1,len(bat_intv))):
            batch = edges[bat_intv[i-1]:bat_intv[i]]
            scores.append(model.decode(emb, batch).squeeze().cpu())

        return pd.DataFrame({
            "protein": [data.data["idx_to_protein"][i] for i in edges[:,0]],
            "chem": [data.data["idx_to_chem"][i] for i in edges[:,1]],
            "score": torch.cat(scores),
            "truth": truth
        }).sort_values("score", ascending=False)

def format_metric_stats(metric_means, metric_stds, split="", std_mult=1.96):
    """Format aggregated summary for logging."""

    splitstr = f"{split}_" if len(split) > 0 else ""
    return " ".join([
        "{}{}: {:.4f}±{:.4f}".format(splitstr, metric, mean, std_mult*metric_stds[metric])
        for metric, mean in metric_means.items()
    ])

def get_train_objs(args):
    Model = LPModel
    Optimizer = torch.optim.Adam
    LRScheduler = torch.optim.lr_scheduler.StepLR
    return Model, Optimizer, LRScheduler

def save_best_model(args, best_model, best_emb):
    np.save(os.path.join(args.save_dir, 'embeddings.npy'),
            best_emb.cpu().detach().numpy())
    if hasattr(best_model.encoder, 'att_adj'):
        filename = os.path.join(args.save_dir, args.dataset + '_att_adj.p')
        pickle.dump(best_model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
        print('Dumped attention adj: ' + filename)

    json.dump(vars(args), open(os.path.join(args.save_dir, 'config.json'), 'w'))
    torch.save(best_model.state_dict(), os.path.join(args.save_dir, 'model.pth'))
    logging.info(f"Saved model in {args.save_dir}")


