import numpy as np
from collections import defaultdict


def ndcg_func(pred, x_test, y_test, top_k_list:list):
    """
    Evaluate nDCG@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_test_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for user_idx in all_user_idx:
        item_idx = all_test_idx[x_test[:, 0] == user_idx]
        pred_u = pred[item_idx]
        true_u = y_test[item_idx]

        for top_k in top_k_list:
            if len(true_u) < top_k:
                break
            log2_iplus1 = np.log2(1+np.arange(1,top_k+1))
            pred_top_k_rel = true_u[np.argsort(-pred_u)][:top_k]
            true_top_k_rel = true_u[np.argsort(-true_u)][:top_k]
            dcg_k = (2**pred_top_k_rel-1) / log2_iplus1
            idcg_k = (2**true_top_k_rel-1) / log2_iplus1

            if np.sum(idcg_k) == 0:
                ndcg_k = 1
            else:
                ndcg_k = np.sum(dcg_k) / np.sum(idcg_k)

            result_map[f"ndcg_{top_k}"].append(ndcg_k)

    return result_map


def recall_func(pred, x_test, y_test, top_k_list:list):
    """
    Evaluate recall@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_test_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for user_idx in all_user_idx:
        item_idx = all_test_idx[x_test[:, 0] == user_idx]
        pred_u = pred[item_idx]
        true_u = y_test[item_idx]
        total_rel = sum(true_u == 1)

        for top_k in top_k_list:
            if len(true_u) < top_k:
                break

            pred_top_k_rel = true_u[np.argsort(-pred_u)][:top_k]
            recall_k = sum(pred_top_k_rel) / total_rel

            if total_rel == 0:
                recall_k = 1.

            result_map[f"recall_{top_k}"].append(recall_k)

    return result_map


def ap_func(pred, x_test, y_test, top_k_list:list):
    """
    Evaluate ap@K of the trained model on test dataset.
    """
    all_user_idx = np.unique(x_test[:,0])
    all_test_idx = np.arange(len(x_test))
    result_map = defaultdict(list)

    for user_idx in all_user_idx:
        item_idx = all_test_idx[x_test[:, 0] == user_idx]
        pred_u = pred[item_idx]
        true_u = y_test[item_idx]

        for top_k in top_k_list:
            if len(true_u) < top_k:
                break
            pred_top_k_rel = true_u[np.argsort(-pred_u)][:top_k]
            N = sum(pred_top_k_rel)

            if N == 0:
                ap_k = 0.
            else:
                precision_k = np.cumsum(pred_top_k_rel) / np.arange(1, top_k+1)
                ap_k = np.sum(precision_k * pred_top_k_rel) / N

            result_map[f"ap_{top_k}"].append(ap_k)

    return result_map
