import os
import numpy as np
import pandas as pd
import re
import pickle
import xarray as xr
from calendar import monthrange
from scripts.xgb_helpers import *
from scripts.xgb_metrics import *
from tqdm.notebook import tqdm
import calendar

from scripts.xgb_helpers import *


def createInputDF_year(Pdf, Ptempprxr):
    """
    Args:
        Pdf (pd.DataFrame): dataframe of yearly stake point MB
        Ptempprxr (pd.DataFrame): dataframe of climate variables (ERA5 or MeteoSuisse)

    Returns:
        pd.DataFrame: Dataframe with each column an input feature at yearly freq, each row is a year
    """
    vars_ = {}
    for i in range(len(Pdf.date_fix0.unique())):
        year = Pdf.date_fix0.iloc[i].year

        # years are from October to Sept of next year
        yearly_range = pd.date_range(str(year) + "-10-01",
                                     str(year + 1) + "-10-01",
                                     freq="1M")

        yearDF = Ptempprxr.sel(time=yearly_range)

        yearDF_prec = yearDF.tp.sum().item(0)
        yearDF_temp = yearDF.t2m.mean().item(0)

        prectemp = [yearDF_prec, yearDF_temp]

        vars_[year + 1] = prectemp

    keys = ["prec", "temp"]
    index = [[var + "_year" for var in keys]]
    inputFeatures = pd.DataFrame(data=vars_).set_index(index).transpose()

    inputFeatures = inputFeatures.rename(columns={
        "prec_year": "tot_precipitation",
        "temp_year": "avg_temperature",
    })
    return inputFeatures


def createInputDF(Pdf,
                  Ptempprxr,
                  input_type,
                  month_val=MONTH_VAL,
                  long_vars=LONG_VARS,
                  match="annual",
                  unseen = False):
    """
    Creates the dataframe used as training input of the XGBoost model.
    Args:
        Pdf (pd.DataFrame): dataframe of yearly stake point MB
        Ptempprxr (xarray): dataframe of climate variables (ERA5 Land or MeteoSuisse)
        input_type (str): indicates if ERA5 Land or MS climate data
        month_val (dic, optional): indices of each month (from 0 October to 12 Sept). Defaults to MONTH_VAL from helpers.
        long_vars (dic, optional): long names of months. Defaults to LONG_VARS from helpers.
        match (str, optional): indicates if match annual, winter or both MB. Defaults to "annual".

    Returns:
        pd.DataFrame:  Dataframe with each column an input feature at monthly freq, each row is a year
    """
    vars_ = {}
    climate_year0 = pd.to_datetime(Ptempprxr["time"].values[0]).year
    climate_year1 = pd.to_datetime(Ptempprxr["time"].values[-1]).year

    stake_year0 = Pdf.date_fix0.iloc[0].year
    stake_year1 = Pdf.date_fix1.iloc[-1].year

    if unseen: 
        begin, end = climate_year0, climate_year1
    else:
        (begin, end) = findOverlapPeriod(stake_year0, stake_year1, climate_year0,
                                     climate_year1)

    for year in range(begin, end):
        yearly_range = pd.date_range(str(year) + "-10-01",
                                     str(year + 1) + "-10-01",
                                     freq="1M")
        winter_range = pd.date_range(str(year) + "-10-01",
                                     str(year + 1) + "-04-01",
                                     freq="1M")
        summer_range = pd.date_range(str(year + 1) + "-04-01",
                                     str(year + 1) + "-10-01",
                                     freq="1M")

        if input_type == 'ERA5-Land':
            yearDF = (Ptempprxr[long_vars.keys()].sel(
                time=yearly_range).to_dataframe().reset_index().drop(
                    ["expver", "latitude", "longitude"], axis=1))
            # Create an input for winter mass balance where summer months are set to 0
            winterDF = (Ptempprxr[long_vars.keys()].sel(
                time=winter_range).to_dataframe().reset_index().drop(
                    ["expver", "latitude", "longitude"], axis=1))
        elif input_type == 'MeteoSuisse':
            yearDF = (Ptempprxr[long_vars.keys()].sel(
                time=yearly_range).to_dataframe().reset_index().drop(
                    ["lat", "lon"], axis=1))
            winterDF = (Ptempprxr[long_vars.keys()].sel(
                time=winter_range).to_dataframe().reset_index().drop(
                    ["lat", "lon"], axis=1))

        # Set the rest of the months in to NaN
        nanarray = np.zeros(6)
        nanarray[:] = np.nan
        restDf = pd.DataFrame(data={
            "time": summer_range,
        })
        for varname in long_vars.keys():
            restDf[varname] = nanarray
        winterDF = pd.concat([winterDF, restDf], axis=0)

        if match == "annual":
            inputDF = yearDF
        else:
            inputDF = winterDF

        # rename columns
        inputDF = inputDF.rename(columns=long_vars)

        allvars = []
        for var in inputDF.columns:
            if var != "time":
                allvars.append(inputDF[var].values)
        vars_[year + 1] = np.concatenate(allvars)

    index = []
    if len(long_vars.keys()) == 1:
        if list(long_vars.keys())[0] == 't2m_corr' or list(
                long_vars.keys())[0] == 'pdd' or list(
                    long_vars.keys())[0] == 't2m':
            index.append([f"t2m_{month_val[i]}" for i in range(1, 13)])
        elif list(long_vars.keys())[0] == 'tp' or list(
                long_vars.keys())[0] == 'tp_corr':
            index.append([f"tp_{month_val[i]}" for i in range(1, 13)])
    else:
        for shortvar in long_vars.keys():
            num_months = 13
            index.append(
                [f"{shortvar}_{month_val[i]}" for i in range(1, num_months)])

    index = np.concatenate(index)
    inputFeatures = pd.DataFrame(data=vars_)
    inputFeatures = inputFeatures.set_index(index).transpose()

    return inputFeatures


def createInputDF_halfyear(Pdf, Ptempprxr):
    """
    Args:
        Pdf (pd.DataFrame): dataframe of yearly stake point MB
        Ptempprxr (xarray): dataframe of climate variables (ERA5 or MeteoSuisse)

    Returns:
        pd.DataFrame: Dataframe with each column an input feature at half-year freq, each row is a year
    """
    vars_ = {}
    for i in range(len(Pdf.date_fix0.unique())):
        year = Pdf.date_fix0.iloc[i].year

        winter_range = pd.date_range(str(year) + "-10-01",
                                     str(year + 1) + "-03-31",
                                     freq="1M")

        summer_range = pd.date_range(str(year + 1) + "-04-01",
                                     str(year + 1) + "-09-30",
                                     freq="1M")

        xr_temppr_why = Ptempprxr.sel(time=winter_range)
        xr_temppr_shy = Ptempprxr.sel(time=summer_range)

        # half year sum and average
        yearDF_why_prec = xr_temppr_why.tp.sum().item(0)
        yearDF_why_temp = xr_temppr_why.t2m.mean().item(0)

        yearDF_shy_prec = xr_temppr_shy.tp.sum().item(0)
        yearDF_shy_temp = xr_temppr_shy.t2m.mean().item(0)

        prectemp = [
            yearDF_why_prec, yearDF_shy_prec, yearDF_why_temp, yearDF_shy_temp
        ]
        vars_[year + 1] = prectemp

    half_years = {1: "WHY", 2: "SHY"}
    index = np.concatenate([
        [f"prec_{half_years[i]}" for i in range(1, 3)],
        [f"temp_{half_years[i]}" for i in range(1, 3)],
    ])
    inputFeatures = pd.DataFrame(data=vars_).set_index(index).transpose()

    return inputFeatures


def createInputDF_seasonal(Pdf, Ptempprxr, input_type):
    """
    Args:
        Pdf (pd.DataFrame): dataframe of yearly stake point MB
        Ptempprxr (xarray): dataframe of climate variables (ERA5 or MeteoSuisse)
        input_type (str): indicates if ERA5 Land or MS climate data

    Returns:
        pd.DataFrame: Dataframe with each column an input feature at seasonal freq, each row is a year
    """
    vars_ = {}
    for i in range(len(Pdf.date_fix0.unique())):
        year = Pdf.date_fix0.iloc[i].year

        yearly_range = pd.date_range(str(year) + "-01-31",
                                     str(year) + "-10-01",
                                     freq="1M")

        xr_temppr_year = Ptempprxr.sel(time=yearly_range)
        # seasonal sum and average
        yearDF_seasonal_prec = xr_temppr_year.tp.resample(time="3M").sum()
        yearDF_seasonal_temp = xr_temppr_year.t2m.resample(time="3M").mean()

        seasonalxr = xr.merge([yearDF_seasonal_prec, yearDF_seasonal_temp])

        if input_type == 'ERA5':
            yearDF = (seasonalxr.to_dataframe().reset_index().drop(
                ["expver", "latitude", "longitude"], axis=1))
        elif input_type == 'MeteoSuisse':
            yearDF = (seasonalxr.to_dataframe().reset_index().drop(
                ["lat", "lon"], axis=1))

        yearDF = yearDF.rename(columns={
            "t2m": "temperature",
            "tp": "precipitation"
        })

        prectemp = yearDF["precipitation"].append(yearDF["temperature"])

        vars_[year + 1] = prectemp
    seasons = {1: "winter", 2: "spring", 3: "summer", 4: "fall"}

    index = np.concatenate([
        [f"prec_{seasons[i]}" for i in range(1, 5)],
        [f"temp_{seasons[i]}" for i in range(1, 5)],
    ])
    inputFeatures = pd.DataFrame(data=vars_).set_index(index).transpose()
    return inputFeatures


def assembleXGStakes(path_save_xgboost_stakes,
                     glStakes_20years_all,
                     rename_stakes,
                     rename = True,
                     multivar=False):
    """Assembles model's predictions for all stakes into one output with all stakes.
    Args:
        path_save_xgboost_stakes (str): path where outputs for each stake were saved by XGBoost model

    Returns:
        dic, dic: (metrics_xg) dictionnary with evaluation metrics of model and (var_xg) dictionnary with outputs of model.
    """
    # Assemble files of all stakes:
    files_pkl = [
        re.split("var_", f)[1][:-4]
        for f in os.listdir(path_save_xgboost_stakes)
        if (f[:3] == "var") and (f != "var_xg.pkl")
    ]

    rmse, mae, correlation, rsquared = {}, {}, {}, {}
    pred_xg = {}
    fold_id, fi_all, mean_fi, feat_test, feat_train = {}, {}, {}, {}, {}
    hp_lr, hp_ne, hp_md = {}, {}, {}
    train_loss, val_loss = {}, {}
    variables = {}
    # for stake in files_pkl:
    for stake in glStakes_20years_all:
        with open(path_save_xgboost_stakes + "var_" + stake + ".pkl",
                  "rb") as fp:
            var_gl = pickle.load(fp)
        stakeName = re.split('_', stake)[0] + '_' + re.split('_', stake)[1]
        if rename: 
            stakeName = rename_stakes[stakeName]
        pred_xg[stakeName] = var_gl["pred_XG"]

        rmse[stakeName] = var_gl["rmse"]
        mae[stakeName] = var_gl["mae"]
        correlation[stakeName] = var_gl["correlation"]
        rsquared[stakeName] = var_gl["rsquared"]

        feat_test[stakeName] = var_gl["feat_test"]
        feat_train[stakeName] = var_gl["feat_train"]
        fold_id[stakeName] = var_gl["fold_id"]
        mean_fi[stakeName] = var_gl["fi_mean"]
        fi_all[stakeName] = var_gl["fi_all"]

        #hyper params
        hp_lr[stakeName] = var_gl["HP_lr"]
        hp_ne[stakeName] = var_gl["HP_ne"]
        hp_md[stakeName] = var_gl["HP_md"]
        train_loss[stakeName] = [
            var_gl["train_loss"][i]['rmse'][-1]
            for i in range(len(var_gl["train_loss"]))
        ]
        val_loss[stakeName] = [
            var_gl["val_loss"][i]['rmse'][-1]
            for i in range(len(var_gl["val_loss"]))
        ]

        # if XGBoost on reduced variable set
        if multivar:
            variables[stakeName] = var_gl['variables']

    metrics_xg = {
        "rmse": rmse,
        "mae": mae,
        "correlation": correlation,
        "rsquared": rsquared,
        "hp_lr": hp_lr,
        "hp_ne": hp_ne,
        "hp_md": hp_md,
        "train_loss": train_loss,
        "val_loss": val_loss
    }

    var_xg = {
        "pred_XG": pred_xg,
        "fold_id": fold_id,
        "mean_fi": mean_fi,
        "all_fi": fi_all,
        "feat_test": feat_test,
        "feat_train": feat_train,
    }
    if multivar:
        var_xg['variables'] = variables
    return var_xg, metrics_xg


def AssembleXGDecades(full_stakes, path_decades):
    decadesfiles = os.listdir(path_decades)
    for stakeName in full_stakes:
        stakeDecades = [
            f for f in decadesfiles if stakeName in f if (f[-8:] != 'full.pkl')
        ]
        rmse, mae, correlation, rsquared, pred_xg = {}, {}, {}, {}, {}
        fold_id, fi_all, mean_fi, feat_test, feat_train = {}, {}, {}, {}, {}
        for decade in stakeDecades:
            with open(path_decades + decade, "rb") as fp:
                #print(f'Opening {decade}')
                var_gl = pickle.load(fp)
                startyear = int(re.split('_', decade)[3])

                pred_xg[startyear] = var_gl["pred_XG"]
                rmse[startyear] = var_gl["rmse"]
                mae[startyear] = var_gl["mae"]
                correlation[startyear] = var_gl["correlation"]
                rsquared[startyear] = var_gl["rsquared"]

                feat_test[startyear] = var_gl["feat_test"]
                feat_train[startyear] = var_gl["feat_train"]
                fold_id[startyear] = var_gl["fold_id"]
                mean_fi[startyear] = var_gl["fi_mean"]
                fi_all[startyear] = var_gl["fi_all"]

        metrics_xg_stake = {
            "rmse": rmse,
            "mae": mae,
            "correlation": correlation,
            "rsquared": rsquared,
        }

        var_xg_stake = {
            "pred_XG": pred_xg,
            "fold_id": fold_id,
            "mean_fi": mean_fi,
            "all_fi": fi_all,
            "feat_test": feat_test,
            "feat_train": feat_train,
        }
        name = f"var_{stakeName}_full.pkl"
        with open(path_decades + name, "wb") as fp:
            pickle.dump(var_xg_stake, fp)
        name = f"metrics_{stakeName}_full.pkl"
        with open(path_decades + name, "wb") as fp:
            pickle.dump(metrics_xg_stake, fp)


def createInputDF_weekly(stakeName, Pdf):
    """
    Args:
        Pdf (pd.DataFrame): dataframe of yearly stake point MB
        stakeName (str): name of stake

    Returns:
        pd.DataFrame: Dataframe with each column an input feature at weekly freq, each row is a year
    """
    vars_ = {}
    long_vars = {"t2m": "temperature", "tp": "precipitation"}

    for i in range(len(Pdf.date_fix0.unique()) - 1):
        year = Pdf.date_fix0.iloc[i].year
        path_xr_weekly = path_ERA5 + "ERA5Land-stakes-hourly/"
        Ptempprxr = xr.open_dataset(path_xr_weekly +
                                    f"{stakeName}_mb_{year}.nc").sortby("time")

        inputDF = (Ptempprxr[["t2m", "tp"]].to_dataframe().reset_index().drop(
            ["latitude", "longitude"], axis=1))
        inputDF = inputDF.rename(columns={
            "t2m": "temperature",
            "tp": "precipitation"
        })

        allvars = []

        for var in inputDF.columns:
            if var != "time":
                # standardize to maximal length:
                diff = 64 - len(inputDF[var].values)
                if diff > 0:
                    diff_array = np.empty(diff)
                    diff_array[:] = np.nan
                    allvars.append(
                        np.concatenate([inputDF[var].values, diff_array]))
                else:
                    allvars.append(inputDF[var].values)
        vars_[year + 1] = np.concatenate(allvars)

    index = []
    for shortvar in long_vars.keys():
        num_weeks = len(allvars[0])
        index.append([f"{shortvar}_week{i}" for i in range(1, num_weeks + 1)])

    index = np.concatenate(index)
    inputFeatures = pd.DataFrame(data=vars_)
    inputFeatures = inputFeatures.set_index(index).transpose()

    return inputFeatures


def getPredGlogem(stake, full_stake, path_glogem, feat_test_gl):
    glogemDF = pd.read_csv(path_glogem + full_stake)

    # create glogem mass balance:
    # cut to past only:
    test_time = feat_test_gl[stake]["time"]
    glogem_time = test_time[test_time < 2017]

    target = feat_test_gl[stake]["target"]
    glogem_target = target[test_time < 2017]
    predictions_gloGEM = glogemDF.set_index(
        "year").loc[glogem_time].smb_ggf.values

    return predictions_gloGEM, glogem_target, glogem_time


def AssemblePDDXG(df_metrics_pdd_a, df_metrics_pdd_aw, df_metrics_pdd_w,
                  df_metrics_monthly_a, df_metrics_monthly_aw,
                  df_metrics_monthly_w):
    """Assembles model metrics from PDD and XGBoost into one DataFrame

    Args:
        df_metrics_pdd_a (pd.DataFrame): PDD with only annual match
        df_metrics_pdd_aw (pd.DataFrame): PDD with winter and annual match
        df_metrics_pdd_w (pd.DataFrame): PDD with only winter match
        df_metrics_monthly_a (pd.DataFrame): XGBoost with only annual match
        df_metrics_monthly_aw (pd.DataFrame): XGBoost with winter and annual match
        df_metrics_monthly_w (pd.DataFrame): XGBoost with only winter match
        df_metrics_lasso (pd.DataFrame): Metrics from linear model
    """

    # Annual and winter match
    df_pdd_grouped_aw = df_metrics_pdd_aw[[
        'glaciers', 'stakes', 'rmse_pdd_a', 'mae_pdd_a', 'nrmse_pdd_a',
        'corr_pdd_a', 'r2_pdd_a', 'nrmse_pdd_a_std', 'mae_pdd_a_std',
        'rmse_pdd_a_std'
    ]].groupby(['glaciers', 'stakes']).mean()

    # Winter metrics:
    df_pdd_grouped_w = df_metrics_pdd_w[[
        'glaciers', 'stakes', 'rmse_pdd_w', 'mae_pdd_w', 'nrmse_pdd_w',
        'corr_pdd_w', 'r2_pdd_w'
    ]].groupby(['glaciers', 'stakes']).mean()

    # Annual match
    df_pdd_grouped_a = df_metrics_pdd_a[[
        'glaciers', 'stakes', 'rmse_pdd_a', 'mae_pdd_a', 'nrmse_pdd_a',
        'corr_pdd_a', 'r2_pdd_a'
    ]].groupby(['glaciers', 'stakes']).mean()

    df_pdd_grouped = df_pdd_grouped_a.merge(df_pdd_grouped_aw,
                                            on=['glaciers', 'stakes'],
                                            suffixes=('', '_aw'))
    df_pdd_grouped = df_pdd_grouped.merge(df_pdd_grouped_w,
                                          on=['glaciers', 'stakes'],
                                          suffixes=('', '_w'))

    # Add XGBoost values:
    df_xg_grouped_a = df_metrics_monthly_a.groupby(['glaciers', 'stakes'
                                                    ]).mean()  # match A
    df_xg_grouped_aw = df_metrics_monthly_aw.groupby(
        ['glaciers', 'stakes']).mean()  # match A and with input W
    df_xg_grouped_w = df_metrics_monthly_w.groupby(['glaciers', 'stakes'
                                                    ]).mean()  # match W

    df_xg_grouped = df_xg_grouped_a.merge(df_xg_grouped_aw,
                                          on=['glaciers', 'stakes'],
                                          suffixes=('', '_aw'))
    df_xg_grouped = df_xg_grouped.merge(df_xg_grouped_w,
                                        on=['glaciers', 'stakes'],
                                        suffixes=('', '_w'))

    # Add XGBoost values:
    df_total_annual = df_pdd_grouped.merge(
        df_xg_grouped, on=['glaciers', 'stakes']).reset_index().merge(
            df_metrics_monthly_a[['glaciers', 'stakes', 'stakes_full']],
            on=['glaciers', 'stakes'])

    # Add differences
    df_total_annual = DiffMetrics(df_total_annual)
    return df_total_annual
