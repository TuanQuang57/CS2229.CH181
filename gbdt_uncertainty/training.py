import json
import os
import numpy as np
import joblib
from gbdt_uncertainty.data import make_train_val_test
import numpy as np
import math
from catboost import Pool, CatBoostRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import catboost as cb


def GP_DoEverything(_train_pool, _val_pool, cd=None, depth=6, seed=0, iters=1000, lr=0.3, iters_rand=100, sigma=0.3, rs=0.1):
    train_pool = [_train_pool[0], _train_pool[1]]
    val_pool = [_val_pool[0], _val_pool[1]]
    tr_pool = cb.Pool(_train_pool[0], 0.00001*np.random.randn(len(_train_pool[1])), cat_features=cd)
    model = cb.CatBoostRegressor(random_seed=seed * iters_rand + 77 * 123, 
                                      iterations=iters_rand, 
                                      learning_rate=0.00001, 
                                      loss_function='RMSE',
                                      bootstrap_type='No',
                                      verbose=False, 
                                      depth=depth,
                                      leaf_estimation_backtracking="No",
                                      leaf_estimation_method="Gradient",
                                      boost_from_average=False, 
                                      random_strength=1e4,
                                      l2_leaf_reg=0,
                                      score_function="L2",
                                      boosting_type='Plain'
    )
    model.fit(tr_pool, use_best_model=False)
    model.save_model("randomtree1.json", format="json", pool=tr_pool)
    with open("randomtree1.json", "r", encoding='utf-8') as f:
        model__ = json.load(f)
        for tree in model__["oblivious_trees"]:
            for ind, (val, weight) in enumerate(zip(tree["leaf_values"], tree["leaf_weights"])):
                    tree["leaf_values"][ind] = np.random.randn() * np.sqrt(len(_train_pool[0])) / np.sqrt(max(1, weight))
        with open("rt1.json", "w") as g:
            json.dump(model__, g)
    model.load_model("rt1.json", format="json")
    s1, b1 = model.get_scale_and_bias()
    model.set_scale_and_bias(s1 * sigma / np.sqrt(iters_rand),  (b1) * sigma / np.sqrt(iters_rand))
    training = cb.Pool(train_pool[0],train_pool[1] - model.predict(train_pool[0]), cat_features=cd) # train_pool[1] - mean_model.predict(train_pool[0]) 
    val = cb.Pool(val_pool[0],val_pool[1] - model.predict(val_pool[0]), cat_features=cd)# val_pool[1] - mean_model.predict(val_pool[0]) 
    model2 = cb.CatBoostRegressor(random_seed=seed * 7 * iters_rand + 3, 
                                  iterations=iters - iters_rand, 
                                  learning_rate=lr, 
                                  loss_function='RMSE',
                                  bootstrap_type='No', 
                                  depth=depth,
                                  leaf_estimation_backtracking="No",
                                  leaf_estimation_method="Gradient",
                                  verbose=False, 
                                  boost_from_average=False,
                                  random_strength=rs,
                                  l2_leaf_reg=0,
                                  score_function="L2",
                                  boosting_type='Plain'
    )
    model2.fit(training, eval_set=val, use_best_model=False)
    model3 = cb.sum_models([model2, model], weights=[1.0, 1.0])
    return model3, np.sqrt(np.mean((val_pool[1] - model3.predict(val_pool[0]))**2))

def tune_parameters_regression(X, y, index_train, index_test, n_splits, alg='sgb'):

    params = []
    seed = 1000 # starting random seed for hyperparameter tuning
    
    for fold in range(n_splits):

        # make catboost pools
        X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test = make_train_val_test(X, y, index_train, index_test, fold)
        full_train_pool = Pool(X_train_all, y_train_all)
        train_pool = Pool(X_train, y_train)
        validation_pool = Pool(X_validation, y_validation)
        test_pool = Pool(X_test, y_test)
        
        # list of hyperparameters for grid search
        # we do not tune the number of trees, it is important for virtual ensembles
        depths = [3, 4, 5, 6] # tree depth
        lrs = [0.001, 0.01, 0.1]# learning rate 
        if alg == "kgb-fixed":
            samples = [0.01, 0.1, 1.0]
        shape = (len(depths), len(lrs), len(samples))

        # perform grid search
        results = np.zeros(shape)
        for d, depth in enumerate(depths):
            for l, lr in enumerate(lrs):
                for s, sample in enumerate(samples):
                    if alg == 'kgb-fixed':
                        model, score = GP_DoEverything([X_train, y_train], [X_validation, y_validation], cd=[],
                                                       seed=seed + 2337 + fold * 113, lr=lr, depth=depth, sigma=0.01, rs=sample)
                        avoid_fitting = True
                    if not avoid_fitting:
                        model.fit(train_pool, eval_set=validation_pool, use_best_model=False)
                    
                    # compute nll
                    results[d, l, s] = score or model.evals_result_['validation']['RMSE'][-1]
                    
                    seed += 1 # update seed
        
        # get best parameters
        argmin = np.unravel_index(np.argmin(results), shape)
        depth = depths[argmin[0]]
        lr = lrs[argmin[1]]
        sample = samples[argmin[2]]
        
        current_params = {'depth': depth, 'lr': lr, 'sample': sample}
        params.append(current_params)
    
    return params
    
def generate_ensemble_regression(dataset_name, X, y, index_train, index_test, n_splits, params, alg="sgb", num_models=10):

    for fold in range(n_splits):

        # make catboost pools
        X_train_all, y_train_all, X_train, y_train, X_validation, y_validation, X_test, y_test = make_train_val_test(X, y, index_train, index_test, fold)
        full_train_pool = Pool(X_train_all, y_train_all)
        test_pool = Pool(X_test, y_test)
    
        # params contains optimal parameters for each fold
        depth = params[fold]['depth']
        lr = params[fold]['lr']
        sample = params[fold]['sample']

        seed = 10 * fold # fix different starting random seeds for all folds
        for i in range(num_models):
            if alg == 'kgb-fixed':
                model, score = GP_DoEverything([X_train, y_train], [X_validation, y_validation], cd=[],
                                                       seed=seed + 359 + i * 13, lr=lr, depth=depth, sigma=0.01, rs=sample)
                avoid_fitting = True
            seed += 1 # new seed for each ensemble element
            if not avoid_fitting:
                model.fit(full_train_pool, eval_set=test_pool, use_best_model=False) # do not use test pool for choosing best iteration
            model.save_model("results/models/" + dataset_name + "_" + alg + "_f" + str(fold) + "_" + str(i), format="cbm")
            

            
