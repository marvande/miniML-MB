import os
import numpy as np
import pandas as pd
import re
import pickle
import xarray as xr
import calendar

from scripts.xgb_helpers import *

def add_extremeYears(Pxr_full, sm_vars, vars_, year = 2022):
    arrays = []
    # Merge on same coordinates as 1960 -> 2021
    first_lat, first_lon = Pxr_full['lat'].to_numpy(), Pxr_full['lon'].to_numpy()
    for i, var in enumerate(vars_):
        if var == "t2m":
            path_nc = path_meteogrid + "/TabsM_verified/lonlat/"
        else:
            path_nc = path_meteogrid + "RhiresM_verified/lonlat/"
        files2022 = [f for f in os.listdir(path_nc) if str(year) in f]
        for month in files2022:
            ds = xr.open_dataset(
                path_nc + month,
                decode_times=False,
            )
            units, reference_date = ds.time.attrs["units"].split("since")
            start_date = pd.to_datetime(reference_date) + pd.DateOffset(
                months=ds.isel(time=0)["time"])
            ds["time"] = pd.date_range(start=start_date,
                                    periods=ds.sizes["time"],
                                    freq="MS")
            ds["lon"] = first_lon
            ds["lat"] = first_lat
            ds = ds.rename_vars({sm_vars[i]: vars_[i]})
            arrays.append(ds)
    Pxr_2022 = xr.merge(arrays).drop("longitude_latitude")
    return Pxr_2022

def createMS_monthly(
    year,
    path_meteogrid,
):
    """Creates monthly xarray for t2m and tp variables.

    Args:
        year (int): 
        path_meteogrid (str): path to Meteo Suisse data

    Returns:
        xarray: monthly MeteoSuisse data for one year with t2m and tp variables
    """
    vars_ = ["t2m", "tp"]
    sm_vars = ["TabsM", "RhiresM"]

    arrays = []
    for i, var in enumerate(vars_):
        if var == "t2m":
            path_nc = path_meteogrid + "/TabsM_verified/lonlat/"
        else:
            path_nc = path_meteogrid + "RhiresM_verified/lonlat/"

        ds = xr.open_dataset(
            path_nc +
            f"{sm_vars[i]}_ch02.lonlat_{year}01010000_{year}12010000.nc",
            decode_times=False,
        )
        units, reference_date = ds.time.attrs["units"].split("since")
        start_date = pd.to_datetime(reference_date) + pd.DateOffset(
            months=ds.isel(time=0)["time"])
        ds["time"] = pd.date_range(start=start_date,
                                   periods=ds.sizes["time"],
                                   freq="MS")

        ds = ds.rename_vars({sm_vars[i]: vars_[i]})
        arrays.append(ds)

    Pxr_month = xr.merge(arrays).drop("longitude_latitude")
    return Pxr_month


def daily_extremeYears(year=2022):
    vars_ = ["t2m", "tp"]
    sm_vars = ["TabsD", "RhiresD"]
    for i, var in enumerate(vars_):
        arrays = []

        # Open reference data for lat/lon coordinates
        if var == "t2m":
            path_nc = path_meteogrid + "/TabsD_verified/lonlat/"
            ds2021 = xr.open_dataset(
                path_nc + 'TabsD_ch02.lonlat_202101010000_202112310000.nc',
                decode_times=False,
            )
        else:
            path_nc = path_meteogrid + "RhiresD_verified/lonlat/"
            ds2021 = xr.open_dataset(
                path_nc + 'RhiresD_ch02.lonlat_202101010000_202112310000.nc',
                decode_times=False,
            )

        first_lat, first_lon = ds2021['lat'].to_numpy(
        ), ds2021['lon'].to_numpy()

        files2022 = [
            f for f in os.listdir(path_nc + 'daily_nc/') if str(year) in f
        ]
        for month in files2022:
            ds = xr.open_dataset(
                path_nc + 'daily_nc/' + month,
                decode_times=False,
            )
            units, reference_date = ds.time.attrs["units"].split("since")

            if month == "TabsD_ch02.lonlat_202301010000_202301310000.nc" or month == "RhiresD_ch02.lonlat_202301010000_202301310000.nc":
                ds["time"] = pd.date_range(start='1/01/2023', end='31/01/2023')
            elif month == "TabsD_ch02.lonlat_202201010000_202201310000.nc" or month == "RhiresD_ch02h.lonlat_202201010000_202201310000.nc":
                ds["time"] = pd.date_range(start='1/01/2022', end='31/01/2022')
            else:
                start_date = pd.to_datetime(reference_date) + pd.DateOffset(
                    months=ds.isel(time=0)["time"])
                ds["time"] = pd.date_range(start=start_date,
                                           periods=ds.sizes["time"],
                                           freq="D")
            ds["lon"] = first_lon
            ds["lat"] = first_lat
            arrays.append(ds)
        Pxr_2023_var = xr.merge(arrays).drop("longitude_latitude")
        name = f"{sm_vars[i]}_ch02.lonlat_{year}01010000_{year}12310000.nc"
        Pxr_2023_var.to_netcdf(path_nc + name)  # save to file
        print('Save', name, 'to :', path_nc)
        
        
def createInputMS(stake_lat, stake_lon, stake_year0, stake_year1, Pxr_full,
                  stakeName, stake_alt, stake_grid_alt, dTdz_stakes, full = False):
    """
    Args:
        stake_lat (float64): latitude for stake
        stake_lon (float64): longitude for stake
        stake_year0 (int): start year of stake MB measurements
        stake_year1 (int): end year of stake MB measurements
        Pxr_full (xarray): xarray of t2m and tp Meteo Suisse variables 

    Returns:
        xarray: MeteoSuisse variables at monthly frequency, 
        at stake location and cut to overlap time of MB measurements
    """
    # Nearest neighbours of MS to MB stake
    Pxr_stake = Pxr_full.sel(lat=stake_lat, lon=stake_lon, method="nearest")

    MS_year0 = pd.to_datetime(Pxr_stake["time"].values[0]).year
    MS_year1 = pd.to_datetime(Pxr_stake["time"].values[-1]).year

    # Change frequency to end of month:
    Pxr_stake = Pxr_stake.resample(time="1M").mean()
    if full:
        begin, end = MS_year0, MS_year1
    else:
        (begin, end) = findOverlapPeriod(stake_year0, stake_year1, MS_year0,
                                     MS_year1)

    # Cut like in MB data:
    Pxr_stake = Pxr_stake.sel(
        time=pd.date_range(str(begin) + "-01-01",
                           str(end) + "-11-01",
                           freq="M"),
        method="nearest",
    )
    # if len(Pxr_stake.lat) > 1:
    #     Pxr_stake = Pxr_stake.isel(lat=1, lon=1)

    # drop last column of 2023 (incomplete data for now)
    Pxr_stake = Pxr_stake.isel(time=slice(0, -1))

    el_stake = stake_alt[stakeName]
    el_grid = stake_grid_alt[stakeName]

    dTdz = dTdz_stakes[stakeName]
    dTdz["time"] = pd.to_datetime(dTdz["time"])
    dTdz = dTdz.set_index("time")
    dtdz = dTdz.dTdz.mean()

    # Correct temperature at stake:
    Pxr_stake['t2m_corr'] = Pxr_stake['t2m'] + (el_stake - el_grid) * dtdz

    # Correct precipitation at stake:
    c_prec = 1.5
    dPdz = 1 / 10000  # % increase/100m constant for now changed to unit/m
    pc = Pxr_stake['tp'] * c_prec
    Pxr_stake['tp_corr'] = pc + pc * ((el_stake - el_grid)) * dPdz
    
    # add PDD (corrected for stake height):
    Pxr_stake = calculatePDD(Pxr_stake, stake_lat, stake_lon, el_stake, el_grid, dtdz)

    return Pxr_stake


def calculatePDD(Pxr_stake, stake_lat, stake_lon, el_stake, el_grid, dtdz):
    path_MS_daily = '../../data/MB_modeling/MeteoSuisse/TabsD_verified/lonlat/'
    # For one year 
    timesteps = pd.to_datetime(Pxr_stake["time"].values)
    pdd_month = []
    for t in timesteps:
        year = t.year
        monthNb = t.month
        num_days = calendar.monthrange(int(year), monthNb)[1]
        
        if monthNb <= 12 and monthNb > 9:
            adj_year = year
        else:
            adj_year = year + 1

        # read daily data for that month
        fileName = f'TabsD_ch02.lonlat_{str(year)}01010000_{str(year)}12310000.nc'
        dailytemp = xr.open_dataset(path_MS_daily + fileName).sel(time=f'{year}-{monthNb}')

        # get at stake location
        if 'dummy' in dailytemp.coords:
            dailytemp_grid = dailytemp.sel(lat=stake_lat, lon=stake_lon,
                                    method="nearest").isel(dummy=0)
        else:
            dailytemp_grid = dailytemp.sel(lat=stake_lat, lon=stake_lon,
                                    method="nearest")

        # correct for stake elevation
        dailytemp_stake = (dailytemp_grid['TabsD'] + (el_stake - el_grid) * dtdz).to_numpy()
        # daily_temps_month.append(dailytemp_stake.to_numpy())
        
        # calculate pdd for that month:
        # check where the daily temperature is actually above 0 deg C
        pos_days = np.where(dailytemp_stake > 0)[0]
        if len(pos_days) > 0:
                pdd = sum(dailytemp_stake[pos_days])  # sum of all positive elements in b
        else:
                pdd = 0
        # reconvert the positive degree-day sum of the total month
        # to an average temperature
        if pdd > 0:
            t_plus = (pdd / num_days)
        else:
            t_plus = 0
            
        pdd_month.append(t_plus)
    Pxr_stake['pdd'] = xr.DataArray(pdd_month, dims='time', coords={'time': Pxr_stake.time})
    
    return Pxr_stake