"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F

from gcn.layers.layers import FermiDiracDecoder
import gcn.manifolds
import gcn.models.encoders as encoders
from gcn.utils.eval_utils import precision_recall_at_k, global_precision_recall_at_k
from gcn.utils.train_utils import default_device

class BaseModel(nn.Module):
    """
    Base model for graph link prediction tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        self.c = args.c

        self.manifold = getattr(gcn.manifolds, self.manifold_name)()
        self.nnodes = args.n_nodes
        self.num_chemicals = args.num_chemicals
        self.num_proteins = args.num_proteins
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def decode(self, h, idx):
        emb_in = h[idx[:, 0], :]
        emb_out = h[idx[:, 1]+self.num_proteins, :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs

    def compute_metrics(self, embeddings, data, split, sample_false=True):
        loss, pos_scores, neg_scores = self.compute_loss_scores(embeddings, data, split, sample_false)
        if pos_scores.is_cuda:
            pos_scores = pos_scores.cpu()
            neg_scores = neg_scores.cpu()
        balanced_labels = [1] * pos_scores.shape[0] + [0] * neg_scores.shape[0]
        balanced_preds = list(pos_scores.detach().numpy()) + list(neg_scores.detach().numpy())
        roc = roc_auc_score(balanced_labels, balanced_preds)
        ap = average_precision_score(balanced_labels, balanced_preds)
        metrics = {'loss': float(loss), 'roc': roc, 'ap': ap}

        if split!='train':
            metrics.update(self.compute_map_metrics(embeddings, data, split))
            for k in [0.5,1]:
                preds_array = np.array(balanced_preds).reshape(-1)
                labels_array = np.array(balanced_labels)
                p_at_k, r_at_k= global_precision_recall_at_k(labels_array,preds_array,k)
                metrics.update({'global_p@{}'.format(k):p_at_k, 'global_r@{}'.format(k):r_at_k})

        return metrics

    def compute_loss_scores(self, embeddings, data, split, sample_false=True):
        edges_false = self.get_edges_false(data, split, sample=sample_false)
        pos_scores = self.decode(embeddings, data.data[f'{split}_edges'])
        neg_scores = self.decode(embeddings, edges_false)
        loss = F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores))
        loss += F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))
        return loss, pos_scores, neg_scores

    def get_edges_false(self, data, split, sample=True):
        edges_false = data.data[f'{split}_edges_false']
        if sample:
            if split == 'train':
                sample_links_idx = np.random.choice(
                    np.arange(0, self.nb_false_edges), self.nb_edges, replace=False)
                edges_false = edges_false[sample_links_idx]
            else:
                n_pos_edges = len(data.data[f'{split}_edges'])
                edges_false = edges_false[:n_pos_edges]
        return edges_false

    def compute_map_metrics(self, embeddings, data, split):
        # for each protein in test/val, generate the scores for all the chemicals in test/val
        protein_idx = np.unique(data.data[f'{split}_edges'][:,0].detach().cpu().numpy())
        chem_idx = np.unique(data.data[f'{split}_edges'][:,1].detach().cpu().numpy())

        chem_order_idx = {item: i for i, item in enumerate(chem_idx)}

        gt = {}
        for k, v in data.data[f'{split}_edges'].detach().cpu().numpy():
            gt.setdefault(k, []).append(chem_order_idx[v])

        gt_matrix = np.zeros((len(protein_idx), len(chem_idx)))
        probs_matrix = np.zeros((len(protein_idx), len(chem_idx)))
        for i in range(len(protein_idx)):
            idx_combinations = np.array(np.meshgrid(protein_idx[i], chem_idx)).T.reshape(-1, 2)
            scores = self.decode(embeddings,idx_combinations).reshape(-1)
            probs_matrix[i] = scores.detach().cpu().numpy()
            gt_matrix[i][gt[protein_idx[i]]] = 1

        at_k_metrics={}
        for topk in [3,5,10]:
            at_k_metrics[f'prec{topk}'],at_k_metrics[f'rec{topk}'] = precision_recall_at_k(gt_matrix, probs_matrix, k=topk)

        return at_k_metrics

    def init_metric_dict(self):
        return {'roc': -1, 'ap': -1}

    def has_improved(self, m1, m2):
        return 0.5 * (m1['roc'] + m1['ap']) < 0.5 * (m2['roc'] + m2['ap'])


class LPModel(BaseModel):
    """
    Base model for link prediction task.
    """

    def __init__(self, args):
        super(LPModel, self).__init__(args)
        self.dc = FermiDiracDecoder(r=args.r, t=args.t)
        self.nb_false_edges = args.nb_false_edges
        self.nb_edges = args.nb_edges
        self.cuda = args.cuda


    def encode(self, x, adj):
        if x is None:
            x = self.embedding.weight

        if not self.cuda == -1:
            adj = adj.to(default_device())
            x = x.to(default_device())

        h = self.encoder.encode(x, adj)
        return h
