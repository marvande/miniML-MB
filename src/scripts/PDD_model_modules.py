import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import re
import xarray as xr
import calendar
from sklearn.linear_model import LinearRegression
from scipy import stats

from scripts.xgb_helpers import *
from scripts.stakes_processing import *

def initArr(t_y):
    # empties array
    return np.empty(len(t_y))

def getSurfaceHeight(glaciers,
                     glStakes,
                     input_type):
    """Get height of climate variable grid and stakes.

    Args:
        glaciers (lsit): list of glaciers
        glStakes (dic): stakes per glacier
        path_glacattr (str, optional): path to stake MB. Defaults to path_glacattr.
        path_ERA5_Land (str, optional): path to ERA5 Land per stake. Defaults to path_ERA5_Land.
    """
    # [m**2 s**-2]
    if input_type == 'ERA5-Land':
        geopot = xr.open_dataset(path_ERA5_Land +
                                 'geo_1279l4_0.1x0.1.grib2_v4_unpack.nc')
    # [m]
    elif input_type == 'MeteoSuisse':
        geopot = xr.open_dataset(
            '../../data/MB_modeling/MeteoSuisse/topo.swiss02_ch02.lonlat.nc',
            decode_times=False)
    else:
        print('Wrong input type. Choose ERA5-Land or MeteoSuisse')

    # get surface height of stake:
    stake_grid_alt, stake_alt = {}, {}
    for gl in glaciers:
        for stake in glStakes[gl]:
            fileName = re.split('.csv', stake)[0]
            # Get coordinates and time of file for this stake:
            df_stake = read_stake_csv(path_glacattr, stake)
            stakeName = re.split('_', fileName)[0] + '_' + re.split(
                '_', fileName)[1]

            # Create corresponding ERA 5 xr
            stake_lat = df_stake.lat.unique()
            stake_lon = df_stake.lon.unique()

            # Height of stake [m]
            stake_alt[stakeName] = df_stake.height.iloc[0]

            # convert to meters:
            if input_type == 'ERA5-Land':
                # get altitude of nearest grid cell:
                geopot_stake = geopot.sel(latitude=stake_lat,
                                          longitude=stake_lon,
                                          method="nearest").isel(
                                              longitude=0, latitude=0,
                                              time=0).z.values
                alt_stake = geopot_stake / (9.80665
                                            )  # Change to [m] from [m^2 s-2]
            elif input_type == 'MeteoSuisse':
                alt_stake = geopot.sel(lat=stake_lat,
                                       lon=stake_lon,
                                       method="nearest").isel(
                                           lon=0, lat=0, time=0).height.values
            stake_grid_alt[stakeName] = float(alt_stake)

    # returns height of era5 land grid and stake
    return stake_grid_alt, stake_alt


def getTemperatureGradients(glaciers,
                            glStakes,
                            path_ERA5=path_ERA5,
                            path_glacattr=path_glacattr):
    """Calculate temperature gradient at stakes.
    Args:
        glaciers (lsit): list of glaciers
        glStakes (dic): stakes per glacier
        path_glacattr (str, optional): path to stake MB. Defaults to path_glacattr.
        path_ERA5_Land (str, optional): path to ERA5 Land per stake. Defaults to path_ERA5_Land.

    Returns:
        dic: temperature gradient at stakes
    """
    # Monthly ERA5 temperature at different pressure levels
    plevelsFile = path_ERA5 + 'era5-monthly-t2m-geopot-plevels.nc'
    xr_plevels = xr.open_dataset(plevelsFile)

    dTdz_stakes = {}
    for gl in tqdm(glaciers, desc='glaciers'):
        for stake in tqdm(glStakes[gl],
                          desc='stakes',
                          leave=False,
                          disable=True):

            # Get coordinates and time of file for this stake:
            fileName = re.split('.csv', stake)[0]
            df_stake = read_stake_csv(path_glacattr, stake)
            stakeName = re.split('_', fileName)[0] + '_' + re.split(
                '_', fileName)[1]

            stake_lat = df_stake.lat.unique()
            stake_lon = df_stake.lon.unique()
            # Create corresponding ERA 5 xr
            # find point closest to stake
            xr_plevels_stake = xr_plevels.sel(latitude=stake_lat,
                                              longitude=stake_lon,
                                              method="nearest").isel(
                                                  longitude=0,
                                                  latitude=0,
                                                  expver=0)

            monthly_slope, monthly_int = [], []
            # from Jan -> Feb
            for t in tqdm(xr_plevels_stake.time[:-2],
                          desc='time',
                          leave=False,
                          disable=True):
                t_plevels = xr_plevels_stake.sel(
                    time=t).to_dataframe().reset_index()[['level', 'z', 't']]
                t_plevels['alt'] = t_plevels['z'] / (
                    9.80665)  # Change to [m] from [m^2 s-2]
                t_plevels[
                    'temp'] = t_plevels['t'] - 273.15  # Change from K to C

                # approximate linear equation between temp and alt:
                reg = LinearRegression().fit(
                    t_plevels['alt'].values.reshape(-1, 1),
                    t_plevels['temp'].values)

                monthly_slope.append(reg.coef_[0])
                monthly_int.append(reg.intercept_)

            df = pd.DataFrame({
                'time': xr_plevels_stake.time[:-2],
                'dTdz': monthly_slope,
                'intercept': monthly_int
            })

            dTdz_stakes[stakeName] = df

    return dTdz_stakes


def getTempGl(stake,
              t_era5,
              dtdz,
              el_stake,
              el_era5,
              year,
              month_pos,
              input_type):
    """Calculates temperature at stake.

    Args:
        stake (str): stake name
        t_era5 (array): temp of era5 at stake location
        dtdz (array): temp gradient of era5 at stake location
        el_stake (float): elevation of stake
        el_era5 (float): elevation of era5
        year (int): year
        month_pos (int): position of month in year (hydr. year)
    """

    # Already corrected to stake height in pre-processing
    stakeName = re.split(".csv", stake)[0][:-3]
    if input_type == 'ERA5-Land':
        df_gl_tg = pd.read_csv(
            f'../../data/MB_modeling/ERA5/ERA5-Land-tg/{stakeName}.csv')
    elif input_type == 'MeteoSuisse':
        df_gl_tg = pd.read_csv(
            f'../../data/MB_modeling/MeteoSuisse/MeteoSuisse-tg/{stakeName}.csv'
        )
    tg = df_gl_tg[(df_gl_tg.year == year)
                    & (df_gl_tg.month_pos == month_pos)].tg.values[0]

    return tg


def getMonthlyTemp(year,
                   month_pos,
                   stake,
                   el_stake,
                   el_era5,
                   dtdz,
                   path_ERA5_Land_hourly=path_ERA5_Land_hourly):

    month_name = list(INVERSE_MONTH_VAL.keys())[month_pos]
    monthNb = INVERSE_MONTH_VAL[month_name]

    if monthNb <= 12 and monthNb > 9:
        adj_year = year
    else:
        adj_year = year + 1

    fileName = f't2m-{str(adj_year)}-{makeStr(monthNb)}.nc'

    # Get hourly temperature
    hourlytemp = xr.open_dataset(path_ERA5_Land_hourly + 't2m/' + fileName)

    # get closest grid cell
    df_stake = read_stake_csv(path_glacattr, stake)
    stake_lat = df_stake.lat.unique()
    stake_lon = df_stake.lon.unique()

    hourlytemp_grid = hourlytemp.sel(latitude=stake_lat,
                                     longitude=stake_lon,
                                     method="nearest").isel(longitude=0,
                                                            latitude=0)

    monthlytemp_grid = hourlytemp_grid.resample(time='1M').mean() - 273

    # correct for glacier height:
    monthlytemp_grid = monthlytemp_grid + (el_stake - el_era5) * dtdz

    # get daily variability and mean
    monthly_std = monthlytemp_grid.std()
    monthly_mean = monthlytemp_grid.mean()

    return monthlytemp_grid, monthly_std.to_array(), monthly_mean.to_array()


def getDailyTemp(year,
                 month_pos,
                 stake,
                 el_stake,
                 el_era5,
                 dtdz,
                 input_type,
                 path_ERA5_Land_hourly=path_ERA5_Land_hourly):

    month_name = list(INVERSE_MONTH_VAL.keys())[month_pos]
    monthNb = INVERSE_MONTH_VAL[month_name]

    if monthNb <= 12 and monthNb > 9:
        adj_year = year
    else:
        adj_year = year + 1

    # get closest grid cell
    df_stake = read_stake_csv(path_glacattr, stake)
    stake_lat = df_stake.lat.unique()
    stake_lon = df_stake.lon.unique()

    # Get hourly temperature
    if input_type == 'ERA5-Land':
        fileName = f't2m-{str(adj_year)}-{makeStr(monthNb)}.nc'
        hourlytemp = xr.open_dataset(path_ERA5_Land_hourly + 't2m/' + fileName)
        hourlytemp_grid = hourlytemp.sel(latitude=stake_lat,
                                         longitude=stake_lon,
                                         method="nearest").isel(longitude=0,
                                                                latitude=0)
        dailytemp_grid = hourlytemp_grid.resample(
            time='1D').mean() - 273  # to degrees from K
    elif input_type == 'MeteoSuisse':
        path_MS_daily = '../../data/MB_modeling/MeteoSuisse/TabsD_verified/lonlat/'
        fileName = f'TabsD_ch02.lonlat_{str(adj_year)}01010000_{str(adj_year)}12310000.nc'
        dailytemp = xr.open_dataset(path_MS_daily + fileName)
        dailytemp_grid = dailytemp.sel(lat=stake_lat,
                                       lon=stake_lon,
                                       method="nearest").isel(lon=0,
                                                              lat=0,
                                                              dummy=0)

    # correct for glacier height:
    dailytemp_grid = dailytemp_grid.sel(time=f'{adj_year}-{monthNb}') + (el_stake - el_era5) * dtdz

    # get daily variability and mean
    daily_std = dailytemp_grid.std()
    daily_mean = dailytemp_grid.mean()

    return dailytemp_grid, daily_std.to_array(), daily_mean.to_array()


def randomDays(mean_month_temp, num_days, std=3):  # degrees in Celsius
    # create fake daily temperature that have same monthly mean as month:
    a, b = mean_month_temp - 5, mean_month_temp + 5
    mu, sigma = mean_month_temp, std**2
    dist = stats.truncnorm((a - mu) / sigma, (b - mu) / sigma,
                           loc=mu,
                           scale=sigma)
    values = dist.rvs(num_days)

    return np.array(values)


def get_PDD(tg,
            year,
            month_pos,
            stake,
            input_type,
            seed=SEED):

    month_name = list(INVERSE_MONTH_VAL.keys())[month_pos]
    monthNb = INVERSE_MONTH_VAL[month_name]

    seed_all(seed)
    num_days = calendar.monthrange(int(year), monthNb)[1]

    # get daily temperature from hourly era5-land data (was preprocessed before)
    stakeName = re.split(".csv", stake)[0][:-3]
    if input_type == 'ERA5-Land':
        df = pd.read_csv(
        f'../../data/MB_modeling/ERA5/ERA5-Land-daily-temp/{stakeName}.csv'
        )
    elif input_type == 'MeteoSuisse':
        df = pd.read_csv(
        f'../../data/MB_modeling/MeteoSuisse/MeteoSuisse-daily-temp/{stakeName}.csv'
        )
    
    daily_temps = df[(df.year == year) & (df.month_pos == month_pos)].drop(
        ['Unnamed: 0', 'year', 'month_pos'], axis=1).values[0]

    # check where the daily temperature is actually above 0 deg C
    pos_days = np.where(daily_temps > 0)[0]

    if len(pos_days) > 0:
        pdd = sum(daily_temps[pos_days])  # sum of all positive elements in b
    else:
        pdd = 0

    # reconvert the positive degree-day sum of the total month
    # to an average temperature
    if pdd > 0:
        t_plus = (pdd / num_days)
    else:
        t_plus = 0

    return t_plus


def getPrecGl(c_prec,
              p_era5,
              dPdz,
              el_stake,
              el_era5,
              tg,
              month,
              T_thresh=1.5):
    # Precipitation at stake:
    pc = p_era5[month] * c_prec

    # extrapolate precipitation with elevation using the gradient dPdZ
    # (stated as % increase / 100m), *elev* contains all
    # elevations on the glacier, *hclim* is the elevation of the
    # ERA-5 gridcell (or the meteo station)

    pg = pc + pc * ((el_stake - el_era5)) * dPdz
    # divide by 100*100 because go from percent to dec and from 100m to 1m

    # determine the state of precipitation:
    # check for elevation bands / grid cells on glacier where
    # local temperature is smaller than the precipitation threshold
    # (typically set to T_thresh=1.5) minus 1

    # all precition is attributed to solid phase, i.e. accumulation (*psg*)
    if tg < T_thresh - 1:
        psg = pg

    # check for elevations in transition phase from 100% solid to
    # 100% liquid precipitation
    elif (tg > T_thresh - 1) & (tg < T_thresh + 1):
        psg = pg * (-(tg - T_thresh - 1.) / 2.)

    else:
        psg = 0

    return pc, pg, psg


def getMelt(
        tg,
        year,
        month,
        sur,
        DDFsnow,  # [m d-1 C-1]
        DDFice,  # [m d-1 C-1]
        density_water,
        T_melt=0,  # [C]
):
    num_days = calendar.monthrange(int(year), month)[1]

    # (*sur*) is snow=1, and the local temperature *tg* in the time
    # step is above a threshold temperature (typically zero deg)
    if (tg > T_melt) & (sur == 1):
        melt_factor = density_water * DDFsnow  # [kg m-3]*[m d-1 C-1]
        monthly_melt = melt_factor * tg * num_days  # [kg m-2 or mm w.e.]

    # type (*sur*) is ice=0, and the local temperature *tg* in
    # the time step is above a threshold temperature (typically zero deg)
    elif (tg > T_melt) & (sur == 0):
        melt_factor = density_water * DDFice  # [kg m-3]*[m d-1 C-1]
        monthly_melt = melt_factor * tg * num_days  # [g m-2 or m w.e.]

    else:
        monthly_melt = 0

    return monthly_melt


def updateSurface(snow, psg, melt):
    # *sno* containing present snow depth at elevation of stake
    # is updated with computed accumulation and
    # melting after every grid step

    snow = snow + psg - melt

    # important: set *sno* back to zero everywhere after one
    # hydrological year (i.e. in October) to prevent cumulation

    # if snow is > 0 the surface type *sur=1*
    if snow > 0:
        sur = 1
    # snow depth cannot be negative, set back
    elif snow < 0:
        snow = 0
        sur = 0
    # if snow is 0 the surface type is ice => *sur=0*
    elif snow == 0:
        sur = 0

    return snow, sur


def getMB(bal, psg, melt):
    # cumulate balances
    # GloGEM also accounts for the component of refreezing but
    # we skip that here

    #bal = bal + density_snow * psg - melt # [m w.e.]
    bal = (bal + psg * 1000 - melt * 1000)  # [m w.e.]

    # note that you store/write out *bal* after every hydrological and
    # set it back to zero. Basically this is a cumulative series and
    # you can extract also winter balance (after e.g. 6 months)

    return round(bal, 4)
