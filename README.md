# CS2229.CH181 - Thuật toán và lý thuyết máy học

|    Thành viên   |    MSSV       |
|-----------------|---------------|
|Nguyễn Kiều Vinh |   18521653    |
|Nguyễn Tuấn Quang|   18521302    |

# Link dataset :

[Year Prediction MSD](https://archive.ics.uci.edu/dataset/203/yearpredictionmsd)
[Scan Localization](https://www.kaggle.com/datasets/uciml/ct-slice-localization)

# Gradient-Boosting-Performs-Gaussian-Process-Inference-Experiments
Under review for ICLR 2023 "Gradient Boosting Performs Gaussian Process Inference"

Implementation of KGB is done in KGB-Experiments/gbdt_uncertainty/training.py and synthetic-regression.ipynb in GP_DoEverything function. 

m -- border_count in CatBoostRegressor, n -- depth in CatBoostRegressor, in GP_DoEverything sigma  -- $\sigma$, itera_rand -- $T_0$, (iters - iters_rand) -- $T_1$, rs (random_strength in CatBoostRegressor) -- $\beta$


Download from UCI YearPredictionMSD and CT Slice Localization data sets, put them into /datasets/ folder

run 'cd KGB-Experiments/gbdt_uncertainty' and then 'python3 generate_odd_regression.py' in order to (re-)generate ood data

To run training run 'cd ..' and then 'python3 train_models.py regression 1'

To get results run 'python3 aggregate_results_regression.py X' where X: std_single, std_ensemble, rmse, prr_auc. 

First two output rmse+std, second outputs only rmse for both signle model and ensemble and the last option outputs PRR for error and AUC-ROC for OOD detection.
