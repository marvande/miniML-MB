import os
import numpy as np
import pandas as pd
import re
import pickle
import xarray as xr

from scripts.xgb_helpers import *


def createInputXR(
    era5landfiles,
    path_ERA5_Land,
    stake_lat,
    stake_lon,
    stake_year0,
    stake_year1,
    variables=["tp", "t2m"],
):
    """Creates the dataframes of ERA5-Land climate variables used as inputs to train XGBoost
        at each stake location (one dataframe per stake).

    Args:
        era5landfiles (list): list of all ERA5-Land files
        path_ERA5_Land (str): path to ERA5-Land files
        stake_lat (float64): latitude for stake
        stake_lon (float64): longitude for stake
        stake_year0 (int): start year of stake MB measurements
        stake_year1 (int): end year of stake MB measurements
        variables (list, optional): list of climate variables to select in small df output. Defaults to ["tp", "t2m"].

    Returns:
        multiple: one dataframe (Ptemppr) at stake location with only variables from above (default t2m, tp), 
        one dataframes with all ERA5 Land variables (Pxr) and the beginning and end years (int) of the overlapping years
        between MB and ERA5 Land measurements.
    """
    arrays = []
    for f in era5landfiles:
        era5_land = xr.open_dataset(path_ERA5_Land + f)
        # Nearest neighbours of ERA5 to MB stake:
        P_ERA5 = era5_land.sel(latitude=stake_lat,
                               longitude=stake_lon,
                               method="nearest")
        era5_year0 = pd.to_datetime(P_ERA5["time"].values[0]).year
        era5_year1 = pd.to_datetime(P_ERA5["time"].values[-1]).year

        (begin, end) = findOverlapPeriod(stake_year0, stake_year1, era5_year0,
                                         era5_year1)

        # Change frequency to end of month:
        # P_ERA5 = P_ERA5.resample(time="1M").interpolate("linear")
        P_ERA5 = P_ERA5.resample(time="1M").mean()

        # Cut like in MB data:
        P_ERA5 = P_ERA5.sel(
            time=pd.date_range(str(begin) + "-01-01",
                               str(end) + "-11-01",
                               freq="M"),
            method="nearest",
        )
        arrays.append(P_ERA5)

    Pxr = xr.merge(arrays)
    era5_year0 = pd.to_datetime(Pxr["time"].values[0]).year
    era5_year1 = pd.to_datetime(Pxr["time"].values[-1]).year

    # Variables that are means but should be sums:
    # total prec:
    # variables_summed = [
    #     "tp", "sd", "sde", "slhf", "ssrd", "sshf", "strd", "ssr", "str"
    # ]
    variables_summed = ["tp"]
    for var in variables_summed:
        Pxr[var] = Pxr[var] * 30

    # Select temperature and precipitation:
    Ptemppr = Pxr[variables].sel(expver=1)
    
    # transform t2m from K to C
    Ptemppr['t2m'] = Ptemppr['t2m'] - 273.15
    # transform tp from m to mm
    Ptemppr['tp'] = Ptemppr['tp'] * 1000

    # In case the dimension of lat and long are bigger than 1:
    if len(Ptemppr.latitude) > 1:
        Ptemppr = Ptemppr.isel(latitude=1, longitude=1)
    return Ptemppr, begin, end
 
 
def createInputXR_weekly(
    year,
    stake_lat,
    stake_lon,
    path_ERA5_Land_hourly,
):
    """Creates the dataframes of ERA5-Land climate variables (t2m, tp and ssr) used as inputs to train XGBoost
        at each stake location at weekly frequency, from hourly ERA5 Land data (one dataframe per stake).

    Args:
        path_ERA5_Land_hourly (str): path to ERA5-Land files
        stake_lat (float64): latitude for stake
        stake_lon (float64): longitude for stake
        
    Returns:
        pd.DataFrame: one dataframe (Ptemppr) at stake location with only variables from above. 
    """
    vars_ = ["t2m", "tp", "ssr"]

    xrmonths = []
    for month in INVERSE_MONTH_VAL.keys():
        monthNb = INVERSE_MONTH_VAL[month]
        if monthNb <= 12 and monthNb > 9:
            adj_year = year
        else:
            adj_year = year + 1
        arrays = []
        for var in vars_:
            fileName = f"{var}-{str(adj_year)}-{makeStr(monthNb)}.nc"

            hourlytemp = xr.open_dataset(path_ERA5_Land_hourly + f"{var}/" +
                                         fileName)

            # Nearest neighbours of ERA5 to MB stake:
            hourlytemp = hourlytemp.sel(latitude=stake_lat,
                                        longitude=stake_lon,
                                        method="nearest")

            arrays.append(hourlytemp)

        Pxr_month = xr.merge(arrays)
        xrmonths.append(Pxr_month)
    Ptemppr = xr.concat(xrmonths, dim="time")
    # Resample to mean per week for some and sum for others
    if var == "t2m":
        Ptemppr = Ptemppr.resample(time="1W").mean()
    else:
        Ptemppr = Ptemppr.resample(time="1W").sum()
    if len(Ptemppr.latitude) > 1:
        Ptemppr = Ptemppr.isel(latitude=1, longitude=1)
        
    # transform t2m from K to C
    Ptemppr['t2m'] = Ptemppr['t2m'] - 273.15
    # transform tp from m to mm
    Ptemppr['tp'] = Ptemppr['tp'] * 1000
    
    return Ptemppr
