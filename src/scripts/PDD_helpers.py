import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import re
import pickle
import xarray as xr

# Scripts:
from scripts.xgb_helpers import *
from scripts.stakes_processing import *
from scripts.PDD_model_modules import *
from scripts.PDD_model_calibration import *
from scripts.xgb_model import *


def dfPDDParams(var_pdd):
    """Creates a dataframe with parameters to fit PDD on stakes. 
       Is used to check chosen parameter range. 
    """
    c_prec, DDFsnow = [], []
    gl_names, stakes, stakes_full = [], [], []
    winter_match, annual_match = [], []

    for key in var_pdd['feat_train'].keys():
        if 'c_prec' in list(var_pdd['feat_train'][key].keys()):
            gl_names.append(key.split("_")[0])
            stakes_full.append(GL_SHORT[key.split("-")[0]] + "-" +
                               key.split("-")[1])
            stakes.append(key.split("-")[1])
            c_prec.append(var_pdd['feat_train'][key]['c_prec'])
            DDFsnow.append(var_pdd['feat_train'][key]['DDFsnow'])
            winter_match.append(var_pdd['feat_train'][key]['winter_match'])
            annual_match.append(var_pdd['feat_train'][key]['annual_match'])

    df_params = pd.DataFrame(
        data={
            "glaciers": gl_names,
            "stakes": stakes,
            "stakes_full": stakes_full,
            "c_prec": c_prec,
            "DDFsnow": DDFsnow,
            'winter_match': winter_match,
            'annual_match': annual_match,
        })
    return df_params


def assemblePDDStakes(path, glStakes_20years_all, rename_stakes):
    """Assembles the results of PDD modelling into two dataframes.

    Args:
        path (str): path to saved PDD results for individual stakes

    Returns:
        2x pd.DataFrame: var_pdd contains predictions and input data, 
        metrics_pdd contains evaluation metrics
    """
    print(f'Reading files from: {path}')
    # Assemble files of all stakes:
    files_pkl = [f[4:-4] for f in os.listdir(path) if (f[:3] == 'var')]

    rmse_a, mae_a, correlation_a, rsquared_a = {}, {}, {}, {}
    rmse_w, mae_w, correlation_w, rsquared_w = {}, {}, {}, {}
    winter_pred_PDD, annual_pred_PDD = {}, {}
    fold_id, feat_test, feat_train = {}, {}, {}

    # for stake in files_pkl:
    for stake in glStakes_20years_all:
        with open(path + 'var_' + stake + '.pkl', 'rb') as fp:
            var_gl = pickle.load(fp)

            # rename stake
            stake = rename_stakes[stake]

            winter_pred_PDD[stake] = var_gl['winter_pred_PDD']
            annual_pred_PDD[stake] = var_gl['annual_pred_PDD']

            rmse_a[stake] = var_gl['rmse_a']
            mae_a[stake] = var_gl['mae_a']
            correlation_a[stake] = var_gl['correlation_a']
            rsquared_a[stake] = var_gl['rsquared_a']

            rmse_w[stake] = var_gl['rmse_w']
            mae_w[stake] = var_gl['mae_w']
            correlation_w[stake] = var_gl['correlation_w']
            rsquared_w[stake] = var_gl['rsquared_w']

            feat_test[stake] = var_gl['feat_test']
            feat_train[stake] = var_gl['feat_train']
            fold_id[stake] = var_gl['fold_id']

    metrics_pdd = {
        'rmse_a': rmse_a,
        'mae_a': mae_a,
        'correlation_a': correlation_a,
        'rsquared_a': rsquared_a,
        'rmse_w': rmse_w,
        'mae_w': mae_w,
        'correlation_w': correlation_w,
        'rsquared_w': rsquared_w,
    }

    var_pdd = {
        'winter_pred_PDD': winter_pred_PDD,
        'annual_pred_PDD': annual_pred_PDD,
        'fold_id': fold_id,
        'feat_test': feat_test,
        'feat_train': feat_train
    }

    # for stake in var_pdd['feat_test'].keys():
    #     var_pdd['feat_test'][stake][
    #         'time'] = var_pdd['feat_test'][stake]['time'] + 1
    #     var_pdd['feat_train'][stake][
    #         'time'] = var_pdd['feat_train'][stake]['time'] + 1

    return var_pdd, metrics_pdd


def findStakesNotProc(
        glStakes_20years,
        path='../../data/MB_modeling/PDD/kfold/match_annual_winter/'):
    st_proc = {}
    for f in os.listdir(path):
        f_split = re.split('_', re.split('.pkl', f)[0])
        if f_split[1] not in st_proc.keys():
            st_proc[f_split[1]] = [f_split[1] + '_' + f_split[2] + '_mb.csv']
        else:
            st_proc[f_split[1]].append(f_split[1] + '_' + f_split[2] +
                                       '_mb.csv')

    rem_stakes = {}
    # add glaciers that have not been processed at all:
    rem_gl = set(st_proc.keys()).symmetric_difference(
        set(glStakes_20years.keys()))
    for gl in rem_gl:
        rem_stakes[gl] = glStakes_20years[gl]

    # add the rest:
    for gl in st_proc.keys():
        diff_ = set(st_proc[gl]).symmetric_difference(set(
            glStakes_20years[gl]))
        if len(diff_) > 0:
            if gl in rem_stakes.keys():
                rem_stakes[gl] = np.concatenate(rem_stakes[gl], diff_)
            else:
                rem_stakes[gl] = [i for i in diff_]

    n_st, n_rem = 0, 0
    for gl in st_proc.keys():
        n_st += len(st_proc[gl])

    for gl in rem_stakes.keys():
        n_rem += len(rem_stakes[gl])

    assert (n_rem + n_st == 28)

    return rem_stakes


def create_prerun_MS(glStakes_20years, path_MS_daily, stake_grid_alt,
                     stake_alt, dTdz_stakes):
    emptyfolder('../../data/MB_modeling/MeteoSuisse/MeteoSuisse-tg/')
    emptyfolder(
        '../../data/MB_modeling/MeteoSuisse/MeteoSuisse-daily-temp/')

    for gl in tqdm(glStakes_20years.keys(), desc="glaciers", position=0):
        print(gl)
        for stakeNb in tqdm(range(len(glStakes_20years[gl])),
                            desc="stakes",
                            leave=False,
                            position=1):
            stake = glStakes_20years[gl][stakeNb]
            stakeName = re.split(".csv", stake)[0][:-3]
            # print(stakeName)
            # Read GLAMOS data:
            df_stake = read_stake_csv(path_glacattr, stake)

            # Read corresponding meteo suisse values for this stake:
            xr_full = xr.open_dataset(path_MS +
                                      f"{stakeName}_mb_full.nc").sortby("time")

            begin_xr = pd.to_datetime(xr_full["time"].values[0]).year
            end_xr = pd.to_datetime(xr_full["time"].values[-1]).year

            # Cut MB data to same years as xr MS:
            df_stake_cut = cutStake(df_stake, begin_xr, end_xr)
            target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]  # Remove cat 0

            el_stake = stake_alt[stakeName]
            el_grid = stake_grid_alt[stakeName]

            # Get years for stake:
            years_stake = [d.year for d in target_DF.date_fix0]
            monthly_range = range(0, 12, 1)

            stake_lat = df_stake.lat.unique()
            stake_lon = df_stake.lon.unique()

            years, months, month_nb, adj_years = [], [], [], []
            tg_monthly, daily_temps_month = [], []

            for year in tqdm(years_stake,
                             position=2,
                             desc='years',
                             disable=True):
                for i in monthly_range:
                    month_name = list(INVERSE_MONTH_VAL.keys())[i]
                    monthNb = INVERSE_MONTH_VAL[month_name]
                    dtdz = dTdz_stakes[stakeName].iloc[
                        i].dTdz  # goes from Jan -> Dec

                    if monthNb <= 12 and monthNb > 9:
                        adj_year = year
                    else:
                        adj_year = year + 1

                    fileName = f'TabsD_ch02.lonlat_{str(adj_year)}01010000_{str(adj_year)}12310000.nc'
                    dailytemp = xr.open_dataset(path_MS_daily + fileName)

                    if 'dummy' in dailytemp.coords:
                        dailytemp_grid = dailytemp.sel(lat=stake_lat,
                                                       lon=stake_lon,
                                                       method="nearest").isel(
                                                           lon=0,
                                                           lat=0,
                                                           dummy=0)
                    else:
                        dailytemp_grid = dailytemp.sel(lat=stake_lat,
                                                       lon=stake_lon,
                                                       method="nearest").isel(
                                                           lon=0, lat=0)
                    monthlytemp_grid = dailytemp_grid.resample(
                        time='1M').mean()  # monthly mean temperature

                    # Select month (to cover hydrological year):
                    one_month_grid = monthlytemp_grid.sel(
                        time=f'{adj_year}-{monthNb}-28', method='nearest')

                    # Correct for glacier height:
                    one_month_stake = one_month_grid + (el_stake -
                                                        el_grid) * dtdz
                    tg_monthly.append(one_month_stake.TabsD.item(0))
                    years.append(year)
                    adj_years.append(adj_year)
                    months.append(i)
                    month_nb.append(monthNb)

                    # Correct daily temp for glacier height:
                    dailytemp_stake = dailytemp_grid.sel(
                        time=f'{adj_year}-{monthNb}') + (el_stake -
                                                         el_grid) * dtdz
                    daily_temps_month.append(dailytemp_stake.TabsD.to_numpy())

            df_monthly_tg = pd.DataFrame(
                data={
                    'year': years,
                    'adj_year': adj_years,
                    'month': month_nb,
                    'month_pos': months,
                    'tg': tg_monthly
                })
            df_monthly_tg.to_csv(
                f'../../data/MB_modeling/MeteoSuisse/MeteoSuisse-tg/{stakeName}.csv'
            )

            df_daily_tg = pd.DataFrame(daily_temps_month)
            df_daily_tg['year'] = years
            df_daily_tg['month_pos'] = months
            df_daily_tg.to_csv(
                f'../../data/MB_modeling/MeteoSuisse/MeteoSuisse-daily-temp/{stakeName}.csv'
            )


def create_prerun_ERA5Land(glStakes_20years, stake_grid_alt, stake_alt,
                           dTdz_stakes):
    emptyfolder('../../data/MB_modeling/ERA5/ERA5-Land-tg/')
    emptyfolder('../../data/MB_modeling/ERA5/ERA5-Land-daily-temp/')

    for gl in tqdm(glStakes_20years.keys(), desc="glaciers", position=0):
        for stakeNb in tqdm(range(len(glStakes_20years[gl])),
                            desc="stakes",
                            leave=False,
                            position=1):
            stake = glStakes_20years[gl][stakeNb]
            stakeName = re.split(".csv", stake)[0][:-3]
            # print(stakeName)
            # Read GLAMOS data:
            df_stake = read_stake_csv(path_glacattr, stake)

            # Read corresponding era 5 land values for this stake:
            xr_full = xr.open_dataset(path_era5_stakes +
                                      f"{stakeName}_mb_full.nc").sortby("time")
            # Cut xr data to start in the 60s like for MeteoSuisse
            xr_full = xr_full.sel(time=slice("1961-01-31",
                                             xr_full.isel(time=-1).time))

            begin_xr = pd.to_datetime(xr_full["time"].values[0]).year
            end_xr = pd.to_datetime(xr_full["time"].values[-1]).year

            # Cut MB data to same years as xr era 5:
            df_stake_cut = cutStake(df_stake, begin_xr, end_xr)
            target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]  # Remove cat 0

            el_stake = stake_alt[stakeName]
            el_grid = stake_grid_alt[stakeName]
            
            

            # Get years for stake:
            years_stake = [d.year for d in target_DF.date_fix0]
            monthly_range = range(0, 12, 1)

            stake_lat = df_stake.lat.unique()
            stake_lon = df_stake.lon.unique()

            years, months, month_nb, adj_years = [], [], [], []
            tg_monthly, daily_temps_month = [], []
            for year in tqdm(years_stake,
                             position=2,
                             desc='years',
                             disable=True):
                for i in monthly_range:
                    month_name = list(INVERSE_MONTH_VAL.keys())[i]
                    monthNb = INVERSE_MONTH_VAL[month_name]
                    dtdz = dTdz_stakes[stakeName].iloc[
                        i].dTdz  # goes from Jan -> Dec

                    df_stake = read_stake_csv(path_glacattr, stake)
                    stake_lat = df_stake.lat.unique()
                    stake_lon = df_stake.lon.unique()

                    if monthNb <= 12 and monthNb > 9:
                        adj_year = year
                    else:
                        adj_year = year + 1

                    # Select month (to cover hydrological year):
                    fileName = f't2m-{str(adj_year)}-{makeStr(monthNb)}.nc'

                    # Get hourly temperature
                    hourlytemp = xr.open_dataset(path_ERA5_Land_hourly +
                                                 't2m/' + fileName)

                    # Get closest grid cell
                    hourlytemp_grid = hourlytemp.sel(latitude=stake_lat,
                                                     longitude=stake_lon,
                                                     method="nearest").isel(
                                                         longitude=0,
                                                         latitude=0)
                    # Monthly t2m:
                    monthlytemp_grid = hourlytemp_grid.resample(
                        time='1M').mean() - 273  # change to degrees from K

                    # Select month (to cover hydrological year):
                    one_month_grid = monthlytemp_grid.sel(
                        time=f'{adj_year}-{monthNb}-28', method='nearest')

                    # Correct for glacier height:
                    one_month_stake = one_month_grid + (el_stake -
                                                        el_grid) * dtdz
                    tg_monthly.append(one_month_stake.t2m.item(0))
                    years.append(year)
                    adj_years.append(adj_year)
                    months.append(i)
                    month_nb.append(monthNb)

                    # Daily t2m:
                    # get daily temperature from hourly era5-land data
                    dailytemp_grid = hourlytemp_grid.resample(
                        time='1D').mean() - 273  # to degrees from K

                    # correct for glacier height:
                    dailytemp_grid = dailytemp_grid.sel(
                        time=f'{adj_year}-{monthNb}') + (el_stake -
                                                         el_grid) * dtdz
                    daily_temps_month.append(dailytemp_grid.t2m.to_numpy())

            df_monthly_tg = pd.DataFrame(
                data={
                    'year': years,
                    'adj_year': adj_years,
                    'month': month_nb,
                    'month_pos': months,
                    'tg': tg_monthly
                })
            df_monthly_tg.to_csv(
                f'../../data/MB_modeling/ERA5/ERA5-Land-tg/{stakeName}.csv')

            df_daily_tg = pd.DataFrame(daily_temps_month)
            df_daily_tg['year'] = years
            df_daily_tg['month_pos'] = months
            df_daily_tg.to_csv(
                f'../../data/MB_modeling/ERA5/ERA5-Land-daily-temp/{stakeName}.csv'
            )
