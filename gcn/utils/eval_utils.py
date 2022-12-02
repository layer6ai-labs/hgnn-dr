import numpy as np


def precision_recall_at_k(actual, predicted, k=10):
    sum_recall = 0.0
    sum_precision = 0.0
    num_users = len(actual)
    true_users = 0
    for act, pred in zip(actual, predicted):
        act_ind_set = set(np.where(act==1)[0])
        pred_ind_set = set(np.argsort(pred)[::-1][:k])
        if len(act_ind_set) != 0:
            sum_recall += len(act_ind_set & pred_ind_set) / float(len(act_ind_set))
            sum_precision += len(act_ind_set & pred_ind_set) / float(k)
            true_users += 1
    assert num_users == true_users
    return sum_precision / true_users, sum_recall / true_users

def global_precision_recall_at_k (actual, predicted, k=1):
    """    Parameters
        ----------
        actual : array, shape = [n_samples]
        Ground truth (true relevance labels).
        predicted :array, shape = [n_samples]
        Predicted scores.
        k : int
        Percentile Rank.
        Returns
        -------
        precision @k : float
    """

    n_samples = len(actual)
    n_topk = round((k/100.)*n_samples)

    act_ind_set = set(np.where(actual==1)[0])
    pred_ind_set = set(np.argsort(predicted)[::-1][:n_topk])

    recall = len(act_ind_set & pred_ind_set) / float(len(act_ind_set))
    precision = len(act_ind_set & pred_ind_set) / float(n_topk)

    return precision, recall


