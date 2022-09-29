# Gradient-Boosting-Performs-Gaussian-Process-Inference-Experiments
Under review for ICLR 2023 "Gradient Boosting Performs Gaussian Process Inference"

Implementation of KGB is done in KGB-Experiments/gbdt_uncertainty/training.py and synthetic-regression.ipynb in GP_DoEverything function. 

m -- border_count in CatBoostRegressor, n -- depth in CatBoostRegressor, in GP_DoEverything sigma  -- $\sigma$, itera_rand -- $T_0$, (iters - iters_rand) -- $T_1$, rs (random_strength in CatBoostRegressor) -- $\beta$


Download from UCI YearPredictionMSD and CT Slice Localization data sets, put them into /datasets/ folder

run 'cd KGB-Experiments/gbdt_uncertainty' and then 'python3 generate_odd_regression.py' in order to (re-)generate ood data

To run training run 'cd ..' and then 'python3 train_models.py regression 1'

To get results run 'python3 aggregate_results_regression.py X' where X: std_single, std_ensemble, rmse, prr_auc. 

First two output rmse+std, second outputs only rmse for both signle model and ensemble and the last option outputs PRR for error and AUC-ROC for OOD detection.
