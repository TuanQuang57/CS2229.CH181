import sys
import json
from gbdt_uncertainty.data import load_regression_dataset
from gbdt_uncertainty.training import tune_parameters_regression, generate_ensemble_regression
import os

mode = sys.argv[1]

def create_dir(name):
    directory = os.path.dirname(name)
    if not os.path.exists(name):
        os.makedirs(name)
    

if mode == "regression":

    try:
        tuning = int(sys.argv[2])
    except:
        print("Tuning parameter is required: 1 if tuning is needed")
        exit(0)
    
    datasets = ["bostonHousing", "concrete", "energy", "kin8nm", 
                "naval-propulsion-plant", "power-plant", "protein-tertiary-structure",
                "wine-quality-red", "yacht", "YearPredictionMSD"]

    algorithms =  ['kgb-fixed']#['sgb-fixed', 'sglb-fixed']#, "kgb-fixed"] 
    # for -fixed we do not tune sample rate and use 0.5 for sbf and 1. for sglb
    
    for name in datasets:
        print("dataset =", name)
    
        if tuning == 1:
            create_dir("results/params")
        
            # Tune hyperparameters
            print("tuning hyperparameters...")
            
            X, y, index_train, index_test, n_splits = load_regression_dataset(name)
            for alg in algorithms:
                print(alg)
                params = tune_parameters_regression(X, y, index_train, 
                                                    index_test, n_splits, alg=alg)
                with open("results/params/" + name + "_" + alg + '.json', 'w') as fp:
                    json.dump(params, fp)
            
        # Training models
        print("training models...")
        create_dir("results/models")
        
        for alg in algorithms:
            print(alg)
            X, y, index_train, index_test, n_splits = load_regression_dataset(name)
            with open("results/params/" + name + "_" + alg + '.json', 'r') as fp:
                params = json.load(fp)
            generate_ensemble_regression(name, X, y, index_train, index_test, 
                                         n_splits, params, alg=alg)
        print()
            
