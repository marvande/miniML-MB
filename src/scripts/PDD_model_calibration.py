import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import re
import pickle
import xarray as xr
from sklearn.model_selection import KFold
import itertools
from random import sample
import random

from scripts.xgb_helpers import *
from scripts.stakes_processing import *
from scripts.PDD_model_modules import *
from scripts.PDD_helpers import *
from scripts.xgb_model import *

# PARAMETERS:
D_SNOW = 0.2
D_ICE = 0.97
D_WATER = 1


# Runs the PDD model over all stakes and saves values for all stakes:
def runPDD_model(
        glStakes_20years,  # dictionary with stakes for each glacier
        dPdz,  # constant
        c_prec,  # constant, initial c_prec
        DDFsnow,  # constant, initial DDFsnow for matching winter MB
        DDFice,  # constant, initial DDFice for matching winter MB
        DDFsnow_range,  # range of DDFsnow for calibration matching annual MB
        c_prec_range,  # range of DDFsnow for calibration matching winter MB
        inital_params,  # initial parameters for PDD model
        seed=SEED,  # seed for reproducibility
        kfold=True,  # run kfold or single fold cross testing
        log=True,  # print log
        match_winter=True,  # match winter MB (otherwise just matches annual MB)
        empty_path=True,  # empty path before saving
        input_type="ERA5-Land",  # input type "ERA5-Land" or "MeteoSuisse"
        calib_style=1,  # 1 for calibrating PDD model, 2 for grid search
):
    if log:
        print('-------------------------------------------------')
        print(f"Running PDD model with params:\n{inital_params}")

    # Get glaciers:
    glaciers = list(glStakes_20years.keys())

    # Get surface height of stakes and era5/MS grid:
    stake_grid_alt, stake_alt = getSurfaceHeight(glaciers,
                                                 glStakes_20years,
                                                 input_type=input_type)

    # Get temperature gradients:
    if log:
        print('Constructing temperature gradients from ERA5:')
    dTdz_stakes = getTemperatureGradients(glaciers, glStakes_20years)

    # Create paths to save model result:
    if kfold:
        fold = "kfold"
    else:
        fold = "single_fold"
    if calib_style == 1:
        if match_winter:
            path = path_pickles + f"{fold}/{input_type}/match_annual_winter/{SEED}/"
        else:
            path = path_pickles + f"{fold}/{input_type}/match_annual/{SEED}/"
    if calib_style == 2:
        path = path_pickles + f"{fold}/{input_type}/calib_2.0/match_annual_winter/"

    # Empty folder and create path if not existing
    createPath(path)
    if empty_path:
        emptyfolder(path)

    if log:
        print(f'Saving pdd predictions to: {path}')

    for gl in tqdm(glaciers, desc="glaciers", position=0):
        for stakeNb in tqdm(range(len(glStakes_20years[gl])),
                            desc="stakes",
                            leave=False,
                            position=1):
            stake = glStakes_20years[gl][stakeNb]
            stakeName = re.split(
                ".csv", stake)[0][:-3]  # stake Name in format Aletsch-P1
            if log:
                print('-------------------------------------------------')
                print(f"Running PDD model for stake: {stakeName}")

            # Read GLAMOS data with PMB for stake:
            df_stake = read_stake_csv(path_glacattr, stake)

            # Get climate data input:
            if input_type == "ERA5-Land":  # (starts in the 50s)
                # Read corresponding era 5 land values for this stake:
                xr_full = xr.open_dataset(path_era5_stakes +
                                          f"{stakeName}_mb_full.nc").sortby(
                                              "time")
                # Cut xr data to start in 1961 like for MeteoSuisse
                xr_full = xr_full.sel(time=slice("1961-01-31",
                                                 xr_full.isel(time=-1).time))

            if input_type == "MeteoSuisse":  # (starts in the 60s)
                # Read corresponding meteo suisse values for this stake:
                xr_full = xr.open_dataset(
                    path_MS + f"{stakeName}_mb_full.nc").sortby("time")

            begin_xr = pd.to_datetime(xr_full["time"].values[0]).year
            end_xr = pd.to_datetime(xr_full["time"].values[-1]).year
            if log:
                print('Beginning and end of PDD model:', begin_xr, end_xr)

            # Cut PMB data to same years as climate data:
            df_stake_cut = cutStake(df_stake, begin_xr, end_xr)

            # Make the index of target_DF its years:
            target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]  # Remove cat 0
            target_years = [d.year for d in target_DF.date_fix1]
            target = target_DF[["date_fix1", "b_a_fix", "b_w_fix"]]
            target["years"] = target_years
            target.set_index("years", inplace=True)
            target = target.loc[target_years]

            # Get elevations of grid and stake:
            el_stake = stake_alt[stakeName]
            el_grid = stake_grid_alt[stakeName]

            # Constants of PDD model:
            constants = {
                "dPdz": dPdz,
                "el_stake": round(el_stake, 3),
                "el_grid": round(el_grid, 3),
            }

            # Run PDD model for stake with 5-fold cross testing:
            (
                winter_pred_PDD,
                annual_pred_PDD,
                feat_test,
                feat_train,
                eval_metrics,
                fold_ids,
            ) = applyPDDModel(stake,
                              xr_full,
                              target,
                              dTdz_stakes,
                              dPdz,
                              c_prec,
                              DDFsnow,
                              DDFice,
                              DDFsnow_range,
                              c_prec_range,
                              inital_params,
                              constants,
                              match_winter=match_winter,
                              input_type=input_type,
                              seed=seed,
                              kfold=kfold,
                              log=log,
                              calib_style=calib_style)

            # Save a pickle for each stake:
            var = {
                "rmse_a": eval_metrics["rmse_a"],
                "mae_a": eval_metrics["mae_a"],
                "correlation_a": eval_metrics["correlation_a"],
                "rsquared_a": eval_metrics["rsquared_a"],
                "rmse_w": eval_metrics["rmse_w"],
                "mae_w": eval_metrics["mae_w"],
                "correlation_w": eval_metrics["correlation_w"],
                "rsquared_w": eval_metrics["rsquared_w"],
                "winter_pred_PDD": winter_pred_PDD,
                "annual_pred_PDD": annual_pred_PDD,
                "fold_id": fold_ids,
                "feat_test": feat_test,
                "feat_train": feat_train,
            }
            with open(path + f"var_{stakeName}.pkl", "wb") as fp:
                pickle.dump(var, fp)


# Applies PDD model for a stake with 5-fold cross testing:
def applyPDDModel(
        stake,  # stake name
        xr_full,  # ERA5 or MeteoSuisse data
        target_DF,  # observed PMB data
        dTdz_stakes,  # temperature gradients for stakes
        dPdz,  # constant
        c_prec,  # initial c_prec
        DDFsnow,  # initial DDFsnow
        DDFice,  # initial DDFice
        DDFsnow_range,  # range of DDFsnow for calibration matching annual MB
        c_prec_range,  # range of DDFsnow for calibration matching winter MB
        inital_params,  # initial parameters for PDD model
        constants,  # constants for PDD model (defined in runPDD_model)
        kfold,  # run kfold or single fold cross testing
        log,  # print log
        match_winter,  # match winter MB (otherwise just matches annual MB)
        input_type,  # input type "ERA5-Land" or "MeteoSuisse"
        seed=SEED,  # seed for reproducibility
        calib_style=1,  # 1 for calibrating PDD model, 2 for grid search
):
    seed_all(seed)

    # Split into training and testing subsets:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)  # 20% split
    if kfold == True:
        numrepets = 5
    else:
        numrepets = 1

    # KFold experiment:
    ann_mb_test_kfold, w_mb_test_kfold, test_time_kfold = ([], [], [])
    ann_mb_train_kfold, w_mb_train_kfold, train_time_kfold = ([], [], [])
    winter_pred_kfold, ann_pred_kfold = [], []
    rmse_w_kfold, mae_w_kfold, pearson_w_kfold, rsquared_w_kfold = [], [], [], []
    rmse_a_kfold, mae_a_kfold, pearson_a_kfold, rsquared_a_kfold = [], [], [], []
    fold_id = []
    c_prec_kfold, DDFsnow_kfold = [], []
    w_matched_kfold, an_matched_kfold = [], []

    # param grid for grid search (calibration style 2)
    param_grid = list(itertools.product(DDFsnow_range, c_prec_range))
    # param_grid = {'DDFsnow': DDFsnow_range, 'c_prec': c_prec_range}

    # For each fold, train and test on remaining fold:
    for i, (train_index, test_index) in enumerate(kf.split(target_DF)):
        # Remove 2022 and 2023 from test index as extreme years:
        test_index = remAdd2022(False,
                                2022 in target_DF.index,
                                target_DF,
                                test_index,
                                year=2022)
        test_index = remAdd2022(False,
                                2023 in target_DF.index,
                                target_DF,
                                test_index,
                                year=2023)

        # Get the annual, winter and time values for test and train:
        ann_mb_test = target_DF["b_a_fix"].iloc[test_index].values  # [mm w.e.]
        ann_mb_train = target_DF["b_a_fix"].iloc[
            train_index].values  # [mm w.e.]
        w_mb_test = target_DF["b_w_fix"].iloc[test_index].values  # [mm w.e.]
        w_mb_train = target_DF["b_w_fix"].iloc[train_index].values  # [mm w.e.]
        # take year to be second part of hydrological year
        test_time = [d.year for d in target_DF.iloc[test_index].date_fix1]
        train_time = [d.year for d in target_DF.iloc[train_index].date_fix1]

        # --------------------------------------------------------------
        # Calibration on training -> search for best PDD parameters:
        # Run PDD model on train years and report best parameters for all years:
        if calib_style == 1:
            c_year, DDFsnow_year, DDFice_year, w_matched_year, an_matched_year = PDDModel_paramsearch(
                stake,
                xr_full,
                train_time,
                ann_mb_train,
                w_mb_train,
                dTdz_stakes,
                dPdz,
                c_prec,
                DDFsnow,
                DDFice,
                DDFsnow_range,
                c_prec_range,
                inital_params,
                constants,
                match_winter,
                input_type=input_type,
                log=log,
            )
            # Take average parameters over training years and make predictions on test years:
            best_c_prec, best_DDF_snow = (
                np.mean(c_year),
                np.mean(DDFsnow_year),
            )

            best_params = {
                'c_prec': best_c_prec,
                'DDFsnow': best_DDF_snow,
                'DDFice': best_DDF_snow * 2
            }

            c_prec_kfold.append(c_year)
            DDFsnow_kfold.append(DDFsnow_year)
            w_matched_kfold.append(w_matched_year)
            an_matched_kfold.append(an_matched_year)

        if calib_style == 2:
            # print('Calibrating with grid search:')
            best_params = calibratePDD2(param_grid, train_time, xr_full,
                                        dTdz_stakes, stake, constants,
                                        input_type, inital_params,
                                        ann_mb_train)

            c_prec_kfold.append(best_params['c_prec'])
            DDFsnow_kfold.append(best_params['DDFsnow'])

        # --------------------------------------------------------------
        # Predictions:
        # Make predictions of winter and annual MB on test years:
        pred_mb_w, pred_mb_ann = PDD_prediction(
            stake,
            xr_full,
            dTdz_stakes,
            test_time,
            best_params,
            constants,
            inital_params,
            input_type=input_type)  # in [mm w.e.]

        # --------------------------------------------------------------
        # Calculate evaluation metrics for each fold:
        rmse_w, mae_w, pearson_w, rsquared2_w = evalMetrics(
            pred_mb_w, w_mb_test)
        rmse_a, mae_a, pearson_a, rsquared2_a = evalMetrics(
            pred_mb_ann, ann_mb_test)

        # Save to KFold experiment (so that can use these later during analysis):
        ann_mb_test_kfold.append(ann_mb_test)
        w_mb_test_kfold.append(w_mb_test)
        test_time_kfold.append(test_time)
        ann_mb_train_kfold.append(ann_mb_train)
        w_mb_train_kfold.append(w_mb_train)
        train_time_kfold.append(train_time)

        winter_pred_kfold.append(pred_mb_w)
        ann_pred_kfold.append(pred_mb_ann)

        rmse_a_kfold.append(rmse_a)
        mae_a_kfold.append(mae_a)
        pearson_a_kfold.append(pearson_a)
        rsquared_a_kfold.append(rsquared2_a)
        rmse_w_kfold.append(rmse_w)
        mae_w_kfold.append(mae_w)
        pearson_w_kfold.append(pearson_w)
        rsquared_w_kfold.append(rsquared2_w)
        fold_id.append(np.tile(i, len(pred_mb_ann)))

        # If only one fold (not 5-fold cross testing)
        if numrepets == 1:
            break  # leave loop after first fold

    # Save all results to a dictionary:
    feat_test = {
        "target_a": np.concatenate(ann_mb_test_kfold),
        "target_w": np.concatenate(w_mb_test_kfold),
        "time": np.concatenate(test_time_kfold),
    }
    if calib_style == 1:
        feat_train = {
            "target_a": np.concatenate(ann_mb_train_kfold),
            "target_w": np.concatenate(w_mb_train_kfold),
            "time": np.concatenate(train_time_kfold),
            'c_prec': np.concatenate(c_prec_kfold),
            'DDFsnow': np.concatenate(DDFsnow_kfold),
            'winter_match': np.concatenate(w_matched_kfold),
            'annual_match': np.concatenate(an_matched_kfold),
        }
    else:
        feat_train = {
            "target_a": np.concatenate(ann_mb_train_kfold),
            "target_w": np.concatenate(w_mb_train_kfold),
            "time": np.concatenate(train_time_kfold),
            'c_prec': c_prec_kfold,
            'DDFsnow': DDFsnow_kfold,
        }
    eval_metrics = {
        "rmse_a": rmse_a_kfold,
        "mae_a": mae_a_kfold,
        "correlation_a": pearson_a_kfold,
        "rsquared_a": rsquared_a_kfold,
        "rmse_w": rmse_w_kfold,
        "mae_w": mae_w_kfold,
        "correlation_w": pearson_w_kfold,
        "rsquared_w": rsquared_w_kfold,
    }

    return (
        np.concatenate(winter_pred_kfold),
        np.concatenate(ann_pred_kfold),
        feat_test,
        feat_train,
        eval_metrics,
        np.concatenate(fold_id),
    )


# Calibration on training -> search for best PDD parameters:
# Runs PDD model on train years and report best parameters for all years:
def PDDModel_paramsearch(
        stake,  # stake name
        xr_full,  # ERA5 or MeteoSuisse data
        train_years,  # training years
        ann_mb_train,  # observed annual MB for training years
        w_mb_train,  # observed winter MB for training years
        dTdz_stakes,  # temperature gradients for stakes
        dPdz,  # constant
        c_prec,  # initial c_prec
        DDFsnow,  # initial DDFsnow
        DDFice,  # initial DDFice
        DDFsnow_range,  # range of DDFsnow for calibration matching annual MB
        c_prec_range,  # range of DDFsnow for calibration matching winter MB
        inital_params,  # initial parameters for PDD model
        constants,  # constants for PDD model (defined in runPDD_model)
        match_winter,  # match winter MB (otherwise just matches annual MB)
        input_type,  # input type "ERA5-Land" or "MeteoSuisse"
        log=False,  # print log
):
    stakeName = re.split(".csv", stake)[0][:-3]

    w_matched_year, an_matched_year = [], []
    c_year, DDFsnow_year, DDFice_year = [], [], []
    # For each training year, calibrate PDD and find best parameters that match observed winter and annual MB:
    for yearNb in tqdm(range(len(train_years)),
                       desc="years",
                       disable=not log,
                       leave=False,
                       position=2):
        year = train_years[yearNb] - 1  # end of hydrological year
        yearly_range = pd.date_range(str(year) + "-10-01",
                                     str(year + 1) + "-10-01",
                                     freq="1M")

        # Get climate variables monthly t2m and tp:
        if input_type == 'ERA5-Land':
            yearDF = (xr_full[[
                "t2m", "tp"
            ]].sel(time=yearly_range).to_dataframe().reset_index().drop(
                ["expver", "latitude", "longitude"], axis=1))
            t_era5 = yearDF["t2m"].values  # [C]
            p_era5 = yearDF["tp"].values / (1000)  # [m]
        elif input_type == 'MeteoSuisse':
            yearDF = (xr_full.sel(
                time=yearly_range).to_dataframe().reset_index().drop(
                    ["lat", "lon"], axis=1))
            t_era5 = yearDF["t2m"].values
            p_era5 = yearDF["tp"].values / (1000)  # [m]

        # Get stake data for that year:
        stake_ann_mb, stake_w_mb = ann_mb_train[yearNb], w_mb_train[yearNb]

        # Constants: temperature gradient for stake and that year
        dTdz = dTdz_stakes[stakeName]
        dTdz["time"] = pd.to_datetime(dTdz["time"])
        dTdz = dTdz.set_index("time")
        dTdzyear = dTdz.loc[str(year)]
        constants["dTdz"] = dTdzyear

        # Calibrate for that year:
        c_prec_best, DDFsnow_best, DDFice_best, winter_matched, annual_matched = calibratePDD(
            stake_w_mb,
            stake_ann_mb,
            stake,
            c_prec,
            c_prec_range,
            DDFsnow_range,
            year,
            t_era5,
            p_era5,
            DDFsnow,
            DDFice,
            constants,
            inital_params,
            log=False,
            match_winter=match_winter,
            input_type=input_type)

        # Save best parameters for that year:
        c_year.append(c_prec_best)
        DDFsnow_year.append(DDFsnow_best)
        DDFice_year.append(DDFice_best)
        w_matched_year.append(winter_matched)
        an_matched_year.append(annual_matched)
    # w_matched_year, an_matched_year indicate whether the PDD was able to match the winter and/or annual MB
    # if False, then the best parameters are the ones that give the closest MB to the target
    return c_year, DDFsnow_year, DDFice_year, w_matched_year, an_matched_year


# Calibrate PDD model for a stake and for a year:
def calibratePDD(
    stake_w_mb,  # observed winter MB for year
    stake_ann_mb,  # observed annual MB for year
    stake,  # stake name
    c_prec,  # initial c_prec
    c_prec_range,  # range of c_prec for calibration matching winter MB
    DDFsnow_range,  # range of DDFsnow for calibration matching annual MB
    year,
    t_era5,  # monthly temperature of year
    p_era5,  # monthly precipitation of year
    DDFsnow,  # initial DDFsnow
    DDFice,  # initial DDFice
    constants,
    inital_params,
    log,
    match_winter,  # match winter MB (otherwise just matches annual MB)
    input_type  # input type "ERA5-Land" or "MeteoSuisse"
):
    # Calibrate c_prec with winter MB:
    if match_winter:
        c_prec_best, winter_matched = matchWinterMB(stake_w_mb, stake,
                                                    c_prec_range, year, t_era5,
                                                    p_era5, DDFsnow, DDFice,
                                                    constants, inital_params,
                                                    input_type, log)
    else:
        # If we don't match winter MB, then the best c_prec are just the initial c_prec
        c_prec_best, winter_matched = c_prec, stake_w_mb

    # Calibrate DDFsnow with annual MB:
    DDFsnow_best, DDFice_best, annual_matched = matchAnnualMB(
        stake_ann_mb, stake, DDFsnow_range, year, t_era5, p_era5, c_prec_best,
        constants, inital_params, input_type, log)

    # Return best parameters for that year:
    return c_prec_best, DDFsnow_best, DDFice_best, winter_matched, annual_matched


# Calibrate c_prec with winter MB:
def matchWinterMB(
    stake_w_mb,  # observed winter MB for year
    stake,  # stake name
    c_prec_range,  # range of c_prec for calibration matching winter MB
    year,
    t_era5,
    p_era5,
    DDFsnow,
    DDFice,
    constants,
    inital_params,
    input_type,
    log=True,
):
    if log:
        print("Initial parameters:")
        print(inital_params)
        print(constants)
        print({"DDFsnow": DDFsnow, "DDFice": DDFice})

    # Threshold under which we consider the match to be good:
    mb_tresh = 0.2 * 1000
    preds = []
    # Test all c_prec values in range:
    for c in tqdm(c_prec_range, desc="c_prec", disable=not log, leave=False):
        yearly_vals = yearlyIt(
            year,
            t_era5,
            p_era5,
            stake,
            c_prec=c,
            DDFsnow=DDFsnow,
            DDFice=DDFice,
            annual=False,
            input_type=input_type,
            **constants,
            **inital_params,
        )  # match winter MB

        # winter mb:
        pred_mb = yearly_vals["mb"][-1]
        preds.append(pred_mb)
        diff_to_target = abs(stake_w_mb - pred_mb)

        # If a good c is found:
        if diff_to_target < mb_tresh:
            return c, True

    # If no good c is found during iteration (none is under the threshold of good MB match)
    # Find the parameter that gives a MB closest to the target:
    df = pd.DataFrame(data={'c': c_prec_range, 'predMB': preds})
    df['diff'] = abs(df['predMB'] - stake_w_mb)
    df = df.sort_values(by='diff', ascending=True)
    second_best_cprec = df['c'].iloc[0]
    best_pred = df['predMB'].iloc[0]
    if log:
        print(
            f"Year:{year}, no winter match, best mb pred: {np.round(best_pred,2)} vs target: {stake_w_mb}\n best cparam {np.round(second_best_cprec,4)}"
        )
    return second_best_cprec, False


# Calibrate DDFsnow with annual MB:
def matchAnnualMB(
    stake_ann_mb,
    stake,
    DDFsnow_range,  # range of DDFsnow for calibration matching annual MB
    year,
    t_era5,
    p_era5,
    c_prec,  # best c_prec from winter match
    constants,
    inital_params,
    input_type,
    log=True,
):
    if log:
        print("Initial parameters:")
        print(inital_params)
        print(constants)
        print({"c_prec": c_prec})

    # Threshold under which we consider the match to be good:
    mb_tresh = 0.2 * 1000
    pred_mbs = []
    for DDFsnow_ in tqdm(DDFsnow_range,
                         desc="DDF_snow",
                         disable=not log,
                         leave=False):
        DDFice_ = DDFsnow_ * 2
        yearly_vals = yearlyIt(
            year,
            t_era5,
            p_era5,
            stake,
            c_prec=c_prec,
            DDFsnow=DDFsnow_,
            DDFice=DDFice_,
            input_type=input_type,
            annual=True,
            **constants,
            **inital_params,
        )  # match annual MB
        # annual mb:
        pred_mb = yearly_vals["mb"][-1]
        pred_mbs.append(pred_mb)

        # If the difference to the target is within the threshold
        diff_to_target = abs(stake_ann_mb - pred_mb)
        if diff_to_target < mb_tresh:
            return DDFsnow_, DDFsnow_ * 2, True

    # If no good DDF is found during iteration (none is under the threshold of good MB match)
    # Find the parameter that gives a MB closest to the target:
    df = pd.DataFrame(data={'ddf': DDFsnow_range, 'predMB': pred_mbs})
    df['diff'] = abs(df['predMB'] - stake_ann_mb)
    df = df.sort_values(by='diff', ascending=True)
    second_best_DDF = df['ddf'].iloc[0]
    best_pred = df['predMB'].iloc[0]
    if log:
        print(
            f"Year:{year}, no annual match, best mb pred: {np.round(best_pred,2)} vs target: {stake_ann_mb}\n best DDFsnow {np.round(second_best_DDF,5)}"
        )
    return second_best_DDF, second_best_DDF * 2, False


# Iterates over the months of a year to match winter MB or annual MB:
def yearlyIt(
    year,  # year number
    t_era5,  # ERA5 temperature of year
    p_era5,  # ERA5 precipitation of year
    stake,  # Stake name
    dTdz,  # Constant
    dPdz,  # Constant
    el_stake,  # Elevation of stake
    el_grid,  # Elevation of era5 grid cell
    c_prec,  # Parameter to tune - prec constant
    DDFsnow,  # Parameter to tune - degree day factor
    DDFice,  # Parameter to tune - degree day factor
    sur0,  # Initial surface type (ice)
    sno0,  # Initial snow depth
    bal0,  # Initial mass balance
    input_type,
    annual=True,
    density_water=D_WATER,
    seed=SEED,
):

    seed_all(seed)

    # Intialise parameters:
    sur = sur0
    snow = sno0
    bal = bal0

    if annual:
        monthly_range = range(0, len(t_era5),
                              1)  # for annual MB (Oct -> Sept) hydr.y
    else:
        monthly_range = range(0, 7, 1)  # for winter MB (Oct ->April/May)

    # Initialise arrays
    tg_year, tg_corr_year = initArr(monthly_range), initArr(monthly_range)
    sno_year, melt_year, sur_year = (
        initArr(monthly_range),
        initArr(monthly_range),
        initArr(monthly_range),
    )
    pera5_year, pg_year, psg_year = (
        initArr(monthly_range),
        initArr(monthly_range),
        initArr(monthly_range),
    )
    bal_year = initArr(monthly_range)

    for i in monthly_range:
        month_name = list(INVERSE_MONTH_VAL.keys())[i]
        monthNb = INVERSE_MONTH_VAL[month_name]

        # Temperature at stake on glacier:
        dtdz = dTdz.iloc[
            i].dTdz  # temperature gradient for month, goes from Jan -> Dec
        tg = getTempGl(stake,
                       t_era5,
                       dtdz,
                       el_stake,
                       el_grid,
                       year,
                       i,
                       input_type=input_type)  # goes from Jan -> Dec

        # Get positive daily average temperature:
        t_plus = get_PDD(
            tg,
            year,
            i,
            stake,
            input_type=input_type,
        )

        # Precipitation and solid precipitation:
        pera5_scaled, pg, psg = getPrecGl(c_prec,
                                          p_era5,
                                          dPdz,
                                          el_stake,
                                          el_grid,
                                          tg,
                                          i,
                                          T_thresh=1.5)
        # Melting:
        melt = getMelt(t_plus,
                       year,
                       monthNb,
                       sur,
                       DDFsnow,
                       DDFice,
                       density_water,
                       T_melt=0)

        # Update surface type
        snow, sur = updateSurface(snow, psg, melt)

        # Calculate mass balance:
        bal = getMB(bal, psg, melt)

        # Update arrays:
        sno_year[i] = snow
        pg_year[i] = pg
        psg_year[i] = psg
        melt_year[i] = melt
        sur_year[i] = sur
        tg_year[i] = tg
        tg_corr_year[i] = t_plus
        bal_year[i] = bal
        pera5_year[i] = pera5_scaled

    yearly_vals = {
        "tg": tg_year,
        "tg_corr": tg_corr_year,
        "pera5_scaled": pera5_year,
        "pg": pg_year,
        "psg": psg_year,
        "snow": sno_year,
        "melt": melt_year,
        "surface": sur_year,
        "mb": bal_year,
    }
    return yearly_vals


# Make predictions on test years:
def PDD_prediction(
    stake,
    xr_full,
    dTdz_stakes,
    test_years,
    best_params,
    constants,
    inital_params,
    input_type,
):
    stakeName = re.split(".csv", stake)[0][:-3]
    pred_mb_w, pred_mb_ann = [], []
    for year in test_years:
        year = year - 1  # start of hydrological year
        yearly_range = pd.date_range(str(year) + "-10-01",
                                     str(year + 1) + "-10-01",
                                     freq="1M")

        # Get climate t2m and tp:
        if input_type == 'ERA5-Land':
            yearDF = (xr_full[[
                "t2m", "tp"
            ]].sel(time=yearly_range).to_dataframe().reset_index().drop(
                ["expver", "latitude", "longitude"], axis=1))
            t_era5 = yearDF["t2m"].values  # [C]
            p_era5 = yearDF["tp"].values / (1000)  # [m]
        elif input_type == 'MeteoSuisse':
            yearDF = (xr_full.sel(
                time=yearly_range).to_dataframe().reset_index().drop(
                    ["lat", "lon"], axis=1))
            t_era5 = yearDF["t2m"].values  # [C]
            p_era5 = yearDF["tp"].values / (1000)  # [m]

        # Constants:
        dTdz = dTdz_stakes[stakeName]
        dTdz["time"] = pd.to_datetime(dTdz["time"])
        dTdz = dTdz.set_index("time")
        constants["dTdz"] = dTdz

        # Calculate predicted winter mb:
        winter_vals = yearlyIt(
            year,
            t_era5,
            p_era5,
            stake,
            c_prec=best_params['c_prec'],
            DDFsnow=best_params['DDFsnow'],
            DDFice=best_params['DDFice'],
            annual=False,
            input_type=input_type,
            **constants,
            **inital_params,
        )
        pred_mb_w.append(winter_vals["mb"][-1])

        # Calculate predicted ammual mb:
        annual_vals = yearlyIt(
            year,
            t_era5,
            p_era5,
            stake,
            c_prec=best_params['c_prec'],
            DDFsnow=best_params['DDFsnow'],
            DDFice=best_params['DDFice'],
            annual=True,
            input_type=input_type,
            **constants,
            **inital_params,
        )
        pred_mb_ann.append(annual_vals["mb"][-1])

    return pred_mb_w, pred_mb_ann


def calibratePDD2(param_grid, train_time, xr_full, dTdz_stakes, stake,
                  constants, input_type, inital_params, ann_mb_train):
    # For each param possibility, run PDD model and get annual mb:
    # train_mae, train_rmse = [], []
    c_sampled, DDF_sampled = [], []
    stakeName = re.split(".csv", stake)[0][:-3]
    
    # Number of random combinations to try
    n_iter = 250
    random_grid = sample(param_grid, n_iter)
    
    best_param, best_error = None, float('inf')
    for _ in range(n_iter):
        # Randomly sample hyperparameters
        DDF_snow_i, c_prec_i = random_grid[_][0], random_grid[_][1]
        c_sampled.append(c_prec_i)
        DDF_sampled.append(DDF_snow_i)
        
        # For each training year, run PDD
        pred_mb_ann = []
        for yearNb in range(len(train_time)):
            year = train_time[yearNb] - 1  # end of hydrological year
            yearly_range = pd.date_range(str(year) + "-10-01",
                                         str(year + 1) + "-10-01",
                                         freq="1M")

            yearDF = (xr_full.sel(
                time=yearly_range).to_dataframe().reset_index().drop(
                    ["lat", "lon"], axis=1))
            t_era5 = yearDF["t2m"].values
            p_era5 = yearDF["tp"].values / (1000)  # [m]

            # Constants: temperature gradient for stake and that year
            dTdz = dTdz_stakes[stakeName]
            dTdz["time"] = pd.to_datetime(dTdz["time"])
            dTdz = dTdz.set_index("time")
            dTdzyear = dTdz.loc[str(year)]
            constants["dTdz"] = dTdzyear

            # Run with DDF_i, c_i
            DDFice_i = DDF_snow_i * 2
            yearly_vals = yearlyIt(
                year,
                t_era5,
                p_era5,
                stake,
                c_prec=c_prec_i,
                DDFsnow=DDF_snow_i,
                DDFice=DDFice_i,
                input_type=input_type,
                annual=True,
                **constants,
                **inital_params,
            )  # match annual MB

            # annual predicted mb:
            pred_mb_ann.append(yearly_vals["mb"][-1])
        # Compare prediction versus observed pmb:
        rmse_a, mae_a, pearson_a, rsquared2_a = evalMetrics(
            pred_mb_ann, ann_mb_train)
        # train_mae.append(mae_a)
        # train_rmse.append(rmse_a)
        if rmse_a < best_error:
            best_error = rmse_a
            best_param = {'DDFsnow': DDF_snow_i, 'DDFice': DDF_snow_i*2, 'c_prec': c_prec_i}
        
    # df_val = pd.DataFrame({'DDFsnow': DDF_sampled, 'c_prec': c_sampled})
    # df_val['train_mae'] = train_mae
    # df_val['train_rmse'] = train_rmse
    # best_DDF_snow = df_val.sort_values(by='train_mae',
    #                                    ascending=True).DDFsnow.values[0]
    # best_c_prec = df_val.sort_values(by='train_mae',
    #                                  ascending=True).c_prec.values[0]
    # best_param = {
    #     'DDFsnow': best_DDF_snow,
    #     'c_prec': best_c_prec,
    #     'DDFice': best_DDF_snow * 2
    # }

    return best_param
