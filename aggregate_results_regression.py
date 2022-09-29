import numpy as np
from catboost import Pool, CatBoostRegressor
from gbdt_uncertainty.data import load_regression_dataset, make_train_val_test
from scipy.stats import ttest_rel
from gbdt_uncertainty.assessment import prr_regression, calc_rmse, ood_detect, ens_rmse
from gbdt_uncertainty.uncertainty import ensemble_uncertainties_regression
import math
import joblib
import sys
from collections import defaultdict

datasets =  ["bostonHousing", "concrete", "energy", "kin8nm", "naval-propulsion-plant",
             "power-plant", "protein-tertiary-structure", "wine-quality-red", "yacht", "YearPredictionMSD"] 
algorithms = ["kgb-fixed"] 

# for proper tables
convert_name = {"weather":"weather","bostonHousing": "BostonH", "concrete": "Concrete", "energy": "Energy", 
                "kin8nm": "Kin8nm", "naval-propulsion-plant": "Naval-p", "power-plant": "Power-p",
                "protein-tertiary-structure": "Protein", "wine-quality-red": "Wine-qu", 
                "yacht": "Yacht", "YearPredictionMSD": "Year"}
     
def load_and_predict(X, name, alg, fold, i):
    model = CatBoostRegressor()
    model.load_model("results/models/" + name + "_" + alg + "_f" + str(fold) + "_" + str(i)) 
    preds = model.predict(X)
    preds = np.array([(p, 1) if isinstance(p, float) else p for p in preds])
    return preds, model
    
def predict(X, model, alg):
    preds = model.predict(X)
    preds = np.array([(p, 1) if isinstance(p, float) else p for p in preds])
    return preds
    
def compute_significance(values_all, metric, minimize=True, raw=False):

    if raw:
        values_all = values_all[:, 0, :]

    values_mean = np.mean(values_all, axis=1) # mean wrt folds or elements
    
    if raw and metric == "rmse":
        values_mean = np.sqrt(values_mean)
    
    # choose best algorithm
    if minimize:
        best_idx = np.nanargmin(values_mean)
    else:
        best_idx = np.nanargmax(values_mean)
        
    textbf = {best_idx} # for all algorithms insignificantly different from the best one
    # compute statistical significance on test or wrt folds

    for idx in range(len(values_mean)):
        test = ttest_rel(values_all[best_idx], values_all[idx]) # paired t-test
        if test[1] > 0.05:
            textbf.add(idx)
            
    return values_mean, textbf

def compute_best(values, minimize=True):

    # choose best algorithm
    if minimize:
        best_idx = np.nanargmin(values)
    else:
        best_idx = np.nanargmax(values)
        
    textbf = {best_idx} 
    for idx in range(len(values)):
        if values[best_idx] == values[idx]: 
            textbf.add(idx)
            
    return textbf
    
def make_table_entry(values_all, metric, minimize=True, round=2, raw=True):
    
    num_values = len(values_all)
    
    values_mean, textbf = compute_significance(values_all, metric, minimize=minimize, raw=raw)

    # prepare all results in latex format

    table = ""

    for idx in range(num_values):
        if idx in textbf:
            table += "\\textbf{" + str(np.round(values_mean[idx], round)) + "} "
        else:    
            table += str(np.round(values_mean[idx], round)) + " "
        table += "& " 
            
    return table
            
def aggregate_results(name, modes = ["single", "ensemble"], 
                      algorithms = ['kgb-fixed'], num_models = 10, 
                      raw=False):
    
    X, y, index_train, index_test, n_splits = load_regression_dataset(name)
    
    results = [] # metric values for all algorithms and all folds
    
    # for ood evaluation
    ood_X_test = np.loadtxt("datasets/ood/" + name)
    if name == "naval-propulsion-plant":
        ood_X_test = ood_X_test[:, :-1]
    ood_size = len(ood_X_test)
        
    for mode in modes:
        for alg in algorithms:
        
            values = defaultdict(lambda: []) # metric values for all folds for given algorithm

            for fold in range(n_splits):
                
                X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test = make_train_val_test(
                                                                                        X, y, index_train, index_test, fold)
                
                
                test_size = len(X_test)
                domain_labels = np.concatenate([np.zeros(test_size), np.ones(ood_size)])

                if mode == "single":
                    preds, model = load_and_predict(X_test, name, alg, fold, 0)
                    values["rmse"].append(calc_rmse(preds[:, 0], y_test, raw=raw))
                    values["KU_prr"].append(float("nan"))
                    values["KU_auc"].append(float("nan"))
                if mode == "ensemble":
                    all_preds = [] # predictions of all models in ensemble
                    all_preds_ood = []
                    
                    for i in range(num_models):
                        preds, model = load_and_predict(X_test, name, alg, fold, i)
                        all_preds.append(preds)
                        preds = predict(ood_X_test, model, alg)
                        all_preds_ood.append(preds)   
                    all_preds = np.array(all_preds)
                    
                    mean_preds = np.mean(all_preds[:, :, 0], axis=0)
                    values["rmse"].append(calc_rmse(mean_preds, y_test, raw=raw))
                    KU = ensemble_uncertainties_regression(np.swapaxes(all_preds, 0, 1))["varm"]


                    values["KU_prr"].append(prr_regression(y_test, mean_preds, KU))
                    
                    all_preds_ood = np.array(all_preds_ood)
                    KU_ood = ensemble_uncertainties_regression(np.swapaxes(all_preds_ood, 0, 1))["varm"]
                    values["KU_auc"].append(ood_detect(domain_labels, KU, KU_ood, mode="ROC"))
                        
            results.append(values)

    return np.array(results)
    
def make_table_element(mean, textbf, idx):
    table = ""
    if np.isnan(mean[idx]):
        table += "--- & "
        return table
    if idx in textbf:
        table += "\\textbf{" + str(int(np.rint(mean[idx]))) + "} "
    else:    
        table += str(int(np.rint(mean[idx]))) + " "
    table += "& "
    return table
                  
table_type = sys.argv[1]
                  
if table_type == "std_single":
                      
    print("===Results with std===") 
    # results with std

    for name in datasets:
        print(name)

        values = aggregate_results(name, modes = ["single"], 
                                   algorithms = ["kgb-fixed"], raw=False)
        
        #print(values)
        #exit(0)
        
        mean = np.mean(values[0]["rmse"])
        std = np.std(values[0]["rmse"])
        print("rmse:", np.round(mean, 2), "$\pm$", np.round(std,2)),

        #mean = np.mean(values[0]["nll"])
        #std = np.std(values[0]["nll"])
        #print("nll:", np.round(mean, 2), "$\pm$", np.round(std,2))
if table_type == "std_ensemble":
                      
    print("===Results with std===") 
    # results with std

    for name in datasets:
        print(name)

        values = aggregate_results(name, modes = ["ensemble"], 
                                   algorithms = ["kgb-fixed"], raw=False)
        
        #print(values)
        #exit(0)
        
        mean = np.mean(values[0]["rmse"])
        std = np.std(values[0]["rmse"])
        print("rmse:", np.round(mean, 2), "$\pm$", np.round(std,2)),

        #mean = np.mean(values[0]["nll"])
        #std = np.std(values[0]["nll"])
        #print("nll:", np.round(mean, 2), "$\pm$", np.round(std,2))
    
if table_type == "rmse":
    print("===NLL and RMSE Table===")
        
    for name in datasets:
        
        raw = False
        if name in ["YearPredictionMSD"]:
            raw = True

        values = aggregate_results(name, raw=raw)
        
        table = convert_name[name] + " & "
        
        values_rmse = np.array([values[i]["rmse"] for i in range(len(values))])
        
        table += make_table_entry(values_rmse, "rmse", round=2, raw=raw)
        print(table.rstrip("& ") + " \\\\")
        
if table_type == "prr_auc":
    print("===PRR and AUC-ROC Table===")
    
    for name in datasets:

        values = aggregate_results(name, modes=["ensemble"], raw=False)
        
        prr_KU = 100*np.array([values[i]["KU_prr"] for i in range(len(values))])

        mean_prr, textbf_prr = compute_significance(prr_KU, "prr", minimize=False)
    
        auc_KU = 100*np.array([values[i]["KU_auc"] for i in range(len(values))])
        mean_auc, textbf_auc = compute_significance(auc_KU, "auc", minimize=False)

        num = len(auc_KU)
    
        table = "" + convert_name[name] + " & "
        
        for idx in range(0, num):
            table += make_table_element(mean_prr, textbf_prr, idx)
            
        for idx in range(0, num):
            table += make_table_element(mean_auc, textbf_auc, idx)
        print(table.rstrip("& ") + " \\\\")
        
        print("\midrule")


