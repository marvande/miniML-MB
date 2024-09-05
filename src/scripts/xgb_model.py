import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import re
import xarray as xr
import re
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV as RSCV
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit  # or StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from scripts.xgb_helpers import *
from scripts.xgb_metrics import *
from scripts.xgb_input import *
from scripts.stakes_processing import *


def runXGBoost(
        glStakes_20years,  # stakes and glaciers
        param_grid,  # grid for HP search
        temporalFreq="annual",  # temporal freq of ERA5 land input
        kfold=True,
        add_2022=False,  # wether to add 2022 in the test data
        mb_match="annual",  # match winter or annual point MB
        path_MS=path_MS,
        path_glacattr=path_glacattr,  # path to glamos data - helpers
        path_era5_stakes=path_era5_stakes,  # path to era5-land at stake - helpers
        month_val=MONTH_VAL,  # values of months (dic) - helpers
        input_vars={
            "t2m": "temperature",
            "tp": "precipitation"
        },
        input_type="ERA5-Land",
        log=False):
    if kfold:
        fold = "kfold"
    else:
        fold = "single_fold"

    vars_ = list(input_vars.keys())[0]
    for var in list(input_vars.keys())[1:]:
        vars_ += "_" + var

    # Create paths to save intermediate result:
    if mb_match == "annual+winter":
        path = (
            path_save_xgboost_stakes +
            f"{fold}/{input_type}/{temporalFreq}/{vars_}/match_annual_winter/")

    elif mb_match == "annual":
        path = path_save_xgboost_stakes + f"{fold}/{input_type}/{temporalFreq}/{vars_}/match_annual/"

    elif mb_match == "winter":
        path = path_save_xgboost_stakes + f"{fold}/{input_type}/{temporalFreq}/{vars_}/match_winter/"

    print(f"Creating path: {path}")

    # Empty folder and create path if non existance
    createPath(path)
    emptyfolder(path)

    # Get list of glaciers
    glaciers = list(glStakes_20years.keys())

    print(f"Matching {mb_match} MB:\n------------------")

    for gl in tqdm(glaciers, desc="glaciers"):
        for stakeNb in tqdm(range(len(glStakes_20years[gl])),
                            desc="stakes",
                            leave=False):

            # Read MB data:
            stake = glStakes_20years[gl][stakeNb]
            stakeName = re.split(".csv", stake)[0][:-3]
            df_stake = read_stake_csv(path_glacattr, stake)

            print(f"Running XGB for stake: {stakeName}\n-------------")

            # Read climate data (pre-processed before to be at location of stake)
            if input_type == "ERA5-Land":
                # Read corresponding era 5 land values for this stake:
                xr_temppr = xr.open_dataset(path_era5_stakes +
                                            f"{stakeName}_mb_full.nc").sortby(
                                                "time")

            if input_type == "MeteoSuisse":
                # Read corresponding meteo suisse values for this stake:
                xr_temppr = xr.open_dataset(
                    path_MS + f"{stakeName}_mb_full.nc").sortby("time")

            begin_xr = pd.to_datetime(xr_temppr["time"].values[0]).year
            end_xr = pd.to_datetime(xr_temppr["time"].values[-1]).year

            # Cut MB data to same years as xr climate:
            df_stake_cut = cutStake(df_stake, begin_xr, end_xr)

            # Remove cat 0 (only modelled MB)
            target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]

            # Create input data:
            if temporalFreq == "annual":
                inputDF = createInputDF_year(df_stake_cut, xr_temppr)
                input_ = {"annual": inputDF}

            elif temporalFreq == "half_year":
                inputDF = createInputDF_halfyear(df_stake_cut, xr_temppr)
                input_ = {"annual": inputDF}

            elif temporalFreq == "seasonal":
                inputDF = createInputDF_seasonal(df_stake_cut, xr_temppr,
                                                 input_type)
                input_ = {"annual": inputDF}

            elif temporalFreq == "monthly":
                inputDF = createInputDF(df_stake_cut,
                                        xr_temppr,
                                        month_val=month_val,
                                        long_vars=input_vars,
                                        input_type=input_type,
                                        match="annual")
                inputDF_winter = createInputDF(df_stake_cut,
                                               xr_temppr,
                                               month_val=month_val,
                                               long_vars=input_vars,
                                               match="winter",
                                               input_type=input_type)
                input_ = {"annual": inputDF, "winter": inputDF_winter}

            elif temporalFreq == "weekly":
                inputDF = createInputDF_weekly(stakeName, df_stake_cut)
                input_ = {"annual": inputDF}

            else:
                print("Error, wrong temporal frequency")

            # Create target:
            target_years = [d.year for d in target_DF.date_fix1
                            ]  # years are end of hydr year.
            target = target_DF[["date_fix1", "b_a_fix", "b_w_fix"]]
            target["years"] = target_years
            target.set_index("years", inplace=True)
            target = target.loc[target_years]

            # Cut inputDF years to same as in target:
            for key in input_.keys():
                intersec = intersection(input_[key].index, target_years)
                input_[key] = input_[key].loc[intersec]
                target = target.loc[intersec]

            if log:
                print('Input DF shape: {}'.format(input_['annual'].shape))
                print(f'Target shape: {target.shape}')
            # Apply XGBoost model
            (predictions_XG, feature_import, fi_all, feat_test, feat_train,
             eval_metrics, fold_ids, validation,
             HP_RF) = applyXGBoost(input_,
                                   target,
                                   add_2022,
                                   kfold,
                                   mb_match,
                                   param_grid,
                                   log=log)

            # Save variables in pkl file
            var = {
                "rmse": eval_metrics["rmse"],
                "mae": eval_metrics["mae"],
                "correlation": eval_metrics["correlation"],
                "rsquared": eval_metrics["rsquared"],
                "pred_XG": predictions_XG,
                "fold_id": fold_ids,
                "feat_test": feat_test,
                "feat_train": feat_train,
                "fi_mean": feature_import,
                "fi_all": fi_all,
                "val_loss": validation["val"],
                "train_loss": validation["train"],
                "epochs": validation["epochs"],
                "best_it": validation["best_it"],
                "HP_lr": HP_RF['learning_rate'],
                "HP_ne": HP_RF['n_estimators'],
                "HP_md": HP_RF['max_depth']
            }
            name = f"var_{stakeName}.pkl"
            with open(path + name, "wb") as fp:
                pickle.dump(var, fp)
                print(f"dictionary {name} saved successfully to {path}")


def applyXGBoost(
        input_,
        target,
        add_2022,
        kfold,
        mb_match,
        param_grid,  # HP grid
        grid_search=True,  # search for best HP
        custom_params_RF=None,  # In case no grid search, can give custom params
        seed=SEED,
        log=False,
        objective='reg:absoluteerror'  # 'reg:squarederror'
):
    seed_all(seed)

    # Get out input data
    inputDF = input_["annual"]

    # Check if 2022 is in the DF
    isthere = 2022 in inputDF.index

    # 20% split for kfold cross-validation
    # (except if size is too small, then do leave one out cv)
    cv = 5
    n_samples = len(target)
    if n_samples <= cv:
        cv = n_samples - 1
    kf = KFold(n_splits=cv, shuffle=True, random_state=seed)

    if kfold == True:
        numrepets = cv
    else:
        numrepets = 1

    # Initialise arrays
    y_test_kfold, time_train_kfold, y_train_kfold, y_val_kfold, time_test_kfold = (
        [],
        [],
        [],
        [],
        [],
    )
    y_trainsub_kfold = []
    pred_XG_kfold, fi_kfold, fold_id = [], [], []
    rmse_kfold, mae_kfold, pearson_kfold, rsquared_kfold = [], [], [], []
    train_loss_kfold, val_loss_kfold, epochs_kfold, best_it_kfold = [], [], [], []

    # rcsv hyperparams:
    lr_kfold, nest_kfold, max_depth_kfold = [], [], []
    x_test_kfold, x_train_kfold, x_val_kfold = [], [], []
    for i, (train2_index, test_index) in enumerate(kf.split(target)):
        # --------------------------------------------------------------------
        # Create input and target arrays:

        # check if we want to remove or add 2022 in the test index
        test_index = remAdd2022(add_2022,
                                2022 in inputDF.index,
                                inputDF,
                                test_index,
                                year=2022)
        test_index = remAdd2022(add_2022,
                                2023 in inputDF.index,
                                inputDF,
                                test_index,
                                year=2023)

        indices = {"test": test_index, "train2": train2_index}

        X, y, time = CreateInputArrays(input_, target, indices, mb_match)

        # Append to KFold experiment:
        y_test_kfold.append(y["test"])
        time_test_kfold.append(time["test"])

        y_train_kfold.append(y["train2"])
        y_trainsub_kfold.append(y["train2"])
        y_val_kfold.append(y["val"])
        time_train_kfold.append(time["train2"])

        # --------------------------------------------------------------------
        # Grid search:

        # If multiple evaluation datasets or multiple evaluation metrics are provided, then early
        # stopping will use the last in the list.
        # scaler = StandardScaler()
        # X_norm = scaler.fit_transform(X['train2'])
        # eval_set = [(scaler.transform(X["train"]), y["train"]), (scaler.transform(X["val"]), y["val"])]
        eval_set = [(X["train"], y["train"]), (X["val"], y["val"])]
        eval_metrics = ["rmse", "mae"]
        if grid_search:
            # return best parameters and best estimator
            xgboost, best_params_, params = RGS(X['train2'],
                                                y["train2"],
                                                eval_set,
                                                param_grid,
                                                eval_metrics,
                                                log_=False,
                                                objective=objective)
            # Fit best model:
            params_RF = {**best_params_, **params}
        else:
            params_RF = custom_params_RF
            params_RF["gpu_id"] = 0
            params_RF["tree_method"] = "gpu_hist"
            params_RF["objective"] = objective
            params_RF["random_state"] = seed
            xgboost = xgb.XGBRegressor(**params_RF)
            xgboost.fit(
                X["train"],
                y["train"],
                eval_metric=eval_metrics,
                eval_set=eval_set,
                verbose=False,
                early_stopping_rounds=10,
            )
        # --------------------------------------------------------------------
        # Evaluate:
        # Best HP rsults:
        num_trees = params_RF["n_estimators"]
        early_stopping_rounds = round(0.1 * num_trees)

        # gets training and validation loss of best estimator:
        results = xgboost.evals_result()
        best_iteration = xgboost.best_iteration
        if log:
            print(
                f"Nb of trees: {num_trees} and early stopping: {early_stopping_rounds}\nBest iteration: {best_iteration}"
            )

        # Save losses:
        val_loss_kfold.append(results["validation_1"])
        train_loss_kfold.append(results["validation_0"])
        epochs_kfold.append(len(results["validation_1"][eval_metrics[0]]))
        best_it_kfold.append(best_iteration)

        # --------------------------------------------------------------------
        # Predictions:
        # Make prediction using the best number of trees from early stopping
        # X_test_norm = scaler.transform(X["test"])
        predictions_XG = xgboost.predict(X["test"],
                                         iteration_range=[0, best_iteration])
        pred_XG_kfold.append(predictions_XG)

        # Feature importance:
        fi_kfold.append(xgboost.feature_importances_)

        # Calculate evaluation metrics:
        rmse, mae, pearson, rsquared2 = evalMetrics(y["test"], predictions_XG)

        # Append them to MC experiment:
        rmse_kfold.append(rmse)
        mae_kfold.append(mae)
        pearson_kfold.append(pearson)
        rsquared_kfold.append(rsquared2)
        fold_id.append(np.tile(i, len(predictions_XG)))

        lr_kfold.append(params_RF['learning_rate'])
        nest_kfold.append(params_RF['n_estimators'])
        max_depth_kfold.append(params_RF['max_depth'])

        x_test_kfold.append(X["test"])
        x_train_kfold.append(X["train"])
        x_val_kfold.append(X["val"])

        if numrepets == 1:
            break

    # Create dictionnaries
    HP_RF = {
        'learning_rate': np.mean(lr_kfold),
        'n_estimators': np.mean(nest_kfold),
        'max_depth': np.mean(max_depth_kfold)
    }

    feat_test = {
        "features": x_test_kfold,
        "target": np.concatenate(y_test_kfold),
        "target_test": y_test_kfold,
        "time": np.concatenate(time_test_kfold),
    }
    feat_train = {
        "features": X["train2"],
        "features_train": x_train_kfold,
        "features_val": x_val_kfold,
        "target": np.concatenate(y_train_kfold),
        "target_val": y_val_kfold,
        "target_train": y_trainsub_kfold,
        "time": np.concatenate(time_train_kfold),
    }

    eval_metrics = {
        "rmse": rmse_kfold,
        "mae": mae_kfold,
        "correlation": pearson_kfold,
        "rsquared": rsquared_kfold,
    }

    validation = {
        "val": val_loss_kfold,
        "train": train_loss_kfold,
        "epochs": epochs_kfold,
        "best_it": best_it_kfold,
    }

    # save mean feature importance over all folds
    mean_fi = pd.DataFrame(fi_kfold).mean(axis=0)
    return (np.concatenate(pred_XG_kfold), mean_fi,
            fi_kfold, feat_test, feat_train, eval_metrics,
            np.concatenate(fold_id), validation, HP_RF)


def CreateInputArrays(input_, target, indices, mb_match):
    # Get out input data
    if mb_match == "annual":
        inputDF = input_["annual"]
    elif mb_match == "annual+winter" or mb_match == "winter":
        inputDF = input_["annual"]
        inputDF_winter = input_["winter"]
    sss = ShuffleSplit(n_splits=1, test_size=0.2)

    if mb_match == "annual":  # match only annual MB
        # Create test and training features
        X_test, y_test, time_test = getXYTime(inputDF,
                                              target,
                                              indices["test"],
                                              type_target="b_a_fix")
        X_train2, y_train2, time_train2 = getXYTime(inputDF,
                                                    target,
                                                    indices["train2"],
                                                    type_target="b_a_fix")

        # Split rest data into train and validation:
        sss.get_n_splits(X_train2, y_train2)
        train_index, val_index = next(sss.split(X_train2, y_train2))
        X_train, y_train, time_train = getXYTime(inputDF,
                                                 target,
                                                 train_index,
                                                 type_target="b_a_fix")
        X_val, y_val, time_val = getXYTime(inputDF,
                                           target,
                                           val_index,
                                           type_target="b_a_fix")

    # Add winter values if needed:
    elif mb_match == "annual+winter":
        # Create annual data:
        X_test, y_test, time_test = getXYTime(inputDF,
                                              target,
                                              indices["test"],
                                              type_target="b_a_fix")
        X_train2, y_train2, time_train2 = getXYTime(inputDF,
                                                    target,
                                                    indices["train2"],
                                                    type_target="b_a_fix")

        sss.get_n_splits(X_train2, y_train2)
        train_index, val_index = next(sss.split(X_train2, y_train2))
        X_train, y_train, time_train = getXYTime(inputDF,
                                                 target,
                                                 train_index,
                                                 type_target="b_a_fix")
        X_val, y_val, time_val = getXYTime(inputDF,
                                           target,
                                           val_index,
                                           type_target="b_a_fix")

        # Create winter data:
        X_train2_w, y_train2_w, time_train2_w = getXYTime(
            inputDF_winter, target, indices["train2"], type_target="b_w_fix")
        X_train_w, y_train_w, time_train_w = getXYTime(inputDF_winter,
                                                       target,
                                                       train_index,
                                                       type_target="b_w_fix")
        X_val_w, y_val_w, time_val_w = getXYTime(inputDF_winter,
                                                 target,
                                                 val_index,
                                                 type_target="b_w_fix")

        # Concatenate winter with annual arrays for training and validation:
        X_train2 = np.concatenate([X_train2, X_train2_w], axis=0)
        y_train2 = np.concatenate([y_train2, y_train2_w], axis=0)
        time_train2 = np.concatenate([time_train2, time_train2_w], axis=0)

        X_train = np.concatenate([X_train, X_train_w], axis=0)
        y_train = np.concatenate([y_train, y_train_w], axis=0)
        time_train = np.concatenate([time_train, time_train_w], axis=0)

        y_val = np.concatenate([y_val, y_val_w], axis=0)
        X_val = np.concatenate([X_val, X_val_w], axis=0)

    elif mb_match == "winter":  # match only winter mass balance
        X_test, y_test, time_test = getXYTime(inputDF_winter,
                                              target,
                                              indices["test"],
                                              type_target="b_w_fix")
        X_train2, y_train2, time_train2 = getXYTime(inputDF_winter,
                                                    target,
                                                    indices["train2"],
                                                    type_target="b_w_fix")

        # Split training into train and validation:
        sss.get_n_splits(X_train2, y_train2)
        train_index, val_index = next(sss.split(X_train2, y_train2))

        X_train, y_train, time_train = getXYTime(inputDF_winter,
                                                 target,
                                                 train_index,
                                                 type_target="b_w_fix")
        X_val, y_val, time_val = getXYTime(inputDF_winter,
                                           target,
                                           val_index,
                                           type_target="b_w_fix")

    X = {"test": X_test, "train2": X_train2, "train": X_train, "val": X_val}
    y = {"test": y_test, "train2": y_train2, "train": y_train, "val": y_val}
    time = {
        "test": time_test,
        "train2": time_train2,
        "train": time_train,
        "val": time_val,
    }

    return X, y, time


def RGS(
        X,
        y,
        eval_set,
        param_grid,
        eval_metrics,
        seed=SEED,
        log_=True,
        model='XGBoost',
        objective='reg:absoluteerror'  #"reg:squarederror"
):
    """Grid search: finds the best hyperparameters

    Args:
        X (np.array): input array
        y (np.array): target array
        eval_set (dic): evaluation set
        param_grid (dic): parameter grid
        eval_metrics (list): evaluation metrics
        seed (float64, optional): Random seed. Defaults to SEED from helpers.
        log_ (bool, optional): if to print. Defaults to True.

    Returns:
        rscv.best_estimator_: best xgboost estimator trained with optimal params, 
        rscv.best_params_: best hyperparameters, 
        params: parameters of model
    """
    params = {"random_state": seed}

    if model == 'XGBoost':
        params["gpu_id"] = 0
        params["tree_method"] = "gpu_hist"
        params["objective"] = objective
        ml_model = xgb.XGBRegressor(**params)

        num_samp = X.shape[0]
        cv = 5
        if num_samp < cv:
            cv = num_samp

        rscv = RSCV(ml_model, param_grid, n_iter=60, cv=cv).fit(
            X,
            y,
            eval_metric=eval_metrics,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=10,
        )

        if log_:
            print(f"Done for {model}!")
            print("Best parameters:\n", rscv.best_params_)
            print("Best score:\n", rscv.best_score_)

        return rscv.best_estimator_, rscv.best_params_, params


def XGBoostDecades(full_stakes,
                   path_glacattr,
                   path_MS,
                   path_decades,
                   input_type,
                   param_grid,
                   kfold,
                   N=1):

    # full 60 years:
    full_years = [
        1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973,
        1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985,
        1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
        1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
        2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021
    ]
    # Cut to decades:
    splits = np.array_split(full_years, 6 / N)

    for stakeName in tqdm(full_stakes, desc='stakes', position=1):
        # Read MB data:
        df_stake = read_stake_csv(path_glacattr, stakeName + '_mb.csv')

        xr_temppr = xr.open_dataset(path_MS +
                                    f"{stakeName}_mb_full.nc").sortby("time")
        begin_xr = pd.to_datetime(xr_temppr["time"].values[0]).year
        end_xr = pd.to_datetime(xr_temppr["time"].values[-1]).year

        # Cut MB data to same years as xr era 5:
        df_stake_cut = cutStake(df_stake, begin_xr, end_xr)

        # Remove cat 0 (only modelled MB)
        target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]

        inputDF = createInputDF(df_stake_cut,
                                xr_temppr,
                                input_type=input_type,
                                long_vars={
                                    't2m': 'temperature',
                                    'tp': 'total precipitation'
                                })
        input_ = {"annual": inputDF}

        # Create target:
        target_years = [d.year for d in target_DF.date_fix1]
        target = target_DF[["date_fix1", "b_a_fix", "b_w_fix"]]
        target["years"] = target_years
        target.set_index("years", inplace=True)
        target = target.loc[target_years]

        # Cut inputDF years to same as in target:
        for key in input_.keys():
            intersec = intersection(input_[key].index, target_years)
            input_[key] = input_[key].loc[intersec]
            target = target.loc[intersec]

        print(f"Running XGB for stake: {stakeName}\n-------------")
        for decade in tqdm(splits, desc='decades', position=2, leave=False):
            #print(f'Running on years: {decade}')

            start = decade[0]
            end = decade[-1]
            target_ = target.reset_index()
            target_dec = target_[target_.years.apply(lambda x: (x <= end) and
                                                     (x >= start))]
            target_dec.set_index(target_dec.years, inplace=True)

            input = input_['annual'].reset_index()
            input = input[input['index'].apply(lambda x: (x <= end) and
                                               (x >= start))]
            input.set_index('index', inplace=True)
            input_dec = {'annual': input}

            #print('Input DF shape: {}'.format(input_dec['annual'].shape))
            #print(f'Target shape: {target_dec.shape}')
            (predictions_XG, feature_import, fi_all, feat_test, feat_train,
             eval_metrics, fold_ids, validation,
             HP_RF) = applyXGBoost(input_dec,
                                   target_dec,
                                   add_2022=False,
                                   kfold=kfold,
                                   mb_match="annual",
                                   param_grid=param_grid,
                                   log=False)
            # Save variables in pkl file
            var = {
                "rmse": eval_metrics["rmse"],
                "mae": eval_metrics["mae"],
                "correlation": eval_metrics["correlation"],
                "rsquared": eval_metrics["rsquared"],
                "pred_XG": predictions_XG,
                "fold_id": fold_ids,
                "feat_test": feat_test,
                "feat_train": feat_train,
                "fi_mean": feature_import,
                "fi_all": fi_all,
                "val_loss": validation["val"],
                "train_loss": validation["train"],
                "epochs": validation["epochs"],
                "best_it": validation["best_it"],
            }
            name = f"var_{stakeName}_{start}_{end}.pkl"
            with open(path_decades + name, "wb") as fp:
                pickle.dump(var, fp)
                #print(
                #    f"dictionary {name} saved successfully to {path_decades}")
