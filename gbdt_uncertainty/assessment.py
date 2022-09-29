from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np

def prr_regression(targets, preds, measure, pos_label=1):
    if pos_label != 1:
        measure_loc = -1.0 * measure
    else:
        measure_loc = measure
    preds = np.squeeze(preds)
    # Compute total MSE
    error = (preds - targets) ** 2
    MSE_0 = np.mean(error)
    # print 'BASE MSE', MSE_0

    # Create array
    array = np.concatenate(
        (preds[:, np.newaxis], targets[:, np.newaxis], error[:, np.newaxis], measure_loc[:, np.newaxis]), axis=1)

    # Results arrays
    results_max = [[0.0, 0.0]]
    results_var = [[0.0, 0.0]]
    results_min = [[0.0, 0.0]]

    optimal_ranking = array[:, 2].argsort()
    sorted_array = array[optimal_ranking]  # Sort by error

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        # Best rejection
        results_max.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])
        # Random Rejection
        results_min.append([float(i) / float(array.shape[0]), float(i) / float(array.shape[0])])

    uncertainty_ranking = array[:, 3].argsort()
    sorted_array = array[uncertainty_ranking]  # Sort by uncertainty

    for i in range(1, array.shape[0]):
        x = np.concatenate((sorted_array[:-i, 0], sorted_array[-i:, 1]), axis=0)
        mse = np.mean((x - sorted_array[:, 1]) ** 2)
        results_var.append([float(i) / float(array.shape[0]), (MSE_0 - mse) / MSE_0])

    max_auc = auc([x[0] for x in results_max], [x[1] for x in results_max])
    var_auc = auc([x[0] for x in results_var], [x[1] for x in results_var])
    min_auc = auc([x[0] for x in results_min], [x[1] for x in results_min])

    AUC_RR = (var_auc - min_auc) / (max_auc - min_auc)

    return AUC_RR

def ood_detect(domain_labels, in_measure, out_measure, mode, pos_label=1):
    scores = np.concatenate((in_measure, out_measure), axis=0)
    scores = np.asarray(scores, dtype=np.float64)
    if pos_label != 1:
        scores *= -1.0

    if mode == 'PR':
        precision, recall, thresholds = precision_recall_curve(domain_labels, scores)
        aupr = auc(recall, precision)
        return aupr

    elif mode == 'ROC':
        roc_auc = roc_auc_score(domain_labels, scores)
        return roc_auc
        
def calc_rmse(preds, target, raw=False):
    if raw:
        return (preds - target)**2 # for individual predictions
    return np.sqrt(np.mean((preds - target)**2))

def ens_rmse(target, preds, epsilon=1e-8, raw=False):
    means = preds[:, :, 0] 
    avg_mean = np.mean(means, axis=0) 
    if raw: # for individual predictions
        return calc_rmse(avg_mean, target, raw=True)
    return calc_rmse(avg_mean, target)

