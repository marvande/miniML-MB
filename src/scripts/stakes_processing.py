from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import re
from itertools import chain, combinations, permutations
import xarray as xr
from datetime import datetime, date, time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from scripts.xgb_helpers import *


# gets all stakes and the number of stakes per glacier
def get_StakesNum(path_GLAMOS_csv):
    onlyfiles = [
        f for f in listdir(path_GLAMOS_csv) if isfile(join(path_GLAMOS_csv, f))
    ]
    glStakesNum, glStakes = {}, {
    }  # number of stakes per glacier and stakes names
    for f in onlyfiles:
        gl = f.split("_")[0]
        if gl not in glStakesNum.keys():
            glStakesNum[gl] = 1
            glStakes[gl] = [f]
        else:
            glStakesNum[gl] = glStakesNum[gl] + 1
            glStakes[gl].append(f)
    return glStakesNum, glStakes


# Reads the dataframe of a stake for one glacier
def read_stake_csv(path, fileName, coi=COI):
    dfStake = pd.read_csv(path + fileName,
                          sep=",",
                          parse_dates=["date_fix0", "date_fix1", "date0", "date1"],
                          header=0).drop(["Unnamed: 0"], axis=1)
    # removes dupl years
    dfStake = dfStake.drop_duplicates()
    dfStake = remove_dupl_years(dfStake).sort_values(by="date_fix0")
    # select only columns of interest
    dfStake = dfStake[coi]
    return dfStake


# Checks for duplicate years for a stake
def remove_dupl_years(df_stake):
    all_years = []
    rows = []
    for row_nb in range(len(df_stake)):
        year = df_stake.date_fix0.iloc[row_nb].year
        if year not in all_years:
            all_years.append(year)
            rows.append(row_nb)
    return df_stake.iloc[rows]


# Checks if there is data missing for a stake
def checkMissingYears(df_stake):
    years_stake = []
    for date in df_stake.date_fix0:
        years_stake.append(date.year)

    missing_years = Diff(
        years_stake,
        [j for j in range(years_stake[0], years_stake[-1] + 1, 1)])
    if len(missing_years) > 0:
        print("Missing years:", missing_years)


# Cuts the dataframe of a stake so that it's the same
# time period as ERA5 data
def cutStake(df_stake, begin_xr, end_xr):
    start_year = df_stake.date_fix0.iloc[0].year
    end_year = df_stake.date_fix1.iloc[-1].year
    print(start_year, end_year)

    offset_begin = abs(begin_xr - start_year)
    offset_end = abs(end_xr - end_year)

    df_stake = df_stake.iloc[offset_begin:len(df_stake) - offset_end]
    return df_stake


# Get stakes with at least N years of data
def getStakesNyears(glaciers,
                    glStakes,
                    path_glacattr,
                    path_era5_stakes,
                    input_type,
                    N=20):
    glStakes_Nyears, glStakesNum_Nyears, glStakes_Nyears_all = {}, {}, []
    for gl in glaciers:
        # One glacier:
        for stake in glStakes[gl]:
            # Get coordinates and time of file for this stake:
            df_stake = read_stake_csv(path_glacattr, stake)

            # Read corresponding era 5 values for this stake:
            stakeName = re.split(".csv", stake)[0][:-3]

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

            # MB data:
            # Check for missing data:
            checkMissingYears(df_stake)

            # Cut MB data to same years as xr era 5:
            df_stake_cut = cutStake(df_stake, begin_xr, end_xr)

            # Remove cat 0
            target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]
            
            # Remove extreme years:
            target_DF = target_DF[target_DF.date_fix0.dt.year < 2021]

            # Keep at least N years:
            if len(target_DF) >= N:
                glStakes_Nyears = updateDic(glStakes_Nyears, gl, stake)
                glStakes_Nyears_all.append(stakeName)

    for gl in glStakes_Nyears.keys():
        glStakesNum_Nyears[gl] = len(glStakes_Nyears[gl])

    glStakes_Nyears_sorted = sorted(glStakesNum_Nyears.items(),
                                    key=lambda x: x[1])
    return glStakes_Nyears, glStakes_Nyears_sorted, glStakes_Nyears_all


# Creates a Dataframe to plot a heatmap for variable var over all stakes
def createHeatMatrixStakes(path_glacattr, glStakes, coi, var="b_a_fix"):
    glaciers = list(glStakes.keys())
    s_end, gl_mb, height = {}, {}, {}
    start_years, end_years = [], []

    # Get values for Heatmap for all stakes
    for g in range(len(glaciers)):
        gl = glaciers[g]  # One glacier
        for stake in glStakes[gl]:
            # Get coordinates and time of file for this stake:
            fileName = re.split(".csv", stake)[0][:-3]
            df_stake = read_stake_csv(path_glacattr, stake,
                                      coi).sort_values(by="date_fix0")

            # remove category 0
            df_stake = df_stake[df_stake.vaw_id > 0]
            
            # remove 2021:
            # df_stake = df_stake[df_stake.date_fix0.dt.year < 2021] 

            # years:
            years = [
                df_stake.date_fix0.iloc[i].year
                for i in range(len(df_stake.date_fix0))
            ]

            start_years.append(years[0])
            end_years.append(years[-1])

            s_end[fileName] = years  # start and end years
            gl_mb[fileName] = df_stake[var].values / (
                1000)  # MB of stake (change to m w.e.)
            height[fileName] = df_stake.height.iloc[0]  # Height of stake

    # Sort stakes per elevation
    stakes_per_el = pd.Series(height).sort_values(ascending=False).index

    # Create DF with MB for each year for all stakes
    totalDF = pd.DataFrame(
        data={
            "years": range(np.min(start_years),
                           np.max(end_years) +
                           1),  # total years over all stakes
            "pres": np.ones((np.max(end_years) + 1) -
                            np.min(start_years)),  # unimportant column
        })

    for stake in stakes_per_el:
        fileName = re.split(".csv", stake)[0]
        year_gl = pd.DataFrame(
            data={
                "years":
                s_end[fileName],  # Years where stake has been measured
                fileName: gl_mb[fileName],  # MB for that stake
            })
        totalDF = pd.merge(totalDF, year_gl, on="years",
                           how="left")  # Add that stake to DF
    # Change years to end of hydrological years
    totalDF["years"] = totalDF["years"] + 1
    totalDF = totalDF.set_index("years").drop(["pres"], axis=1)
    
    # stake elevations:
    el_stakes = pd.Series(height).sort_values(ascending=False)
    return totalDF, el_stakes

# Rename stakes so that ordered by elevation from P1 to P(X highest):
def rename_stakes(glStakes_20years):
    glaciers = list(glStakes_20years.keys())
    s_end, gl_mb, = {}, {}
    start_years, end_years = [], []
    stakes_per_el = {}
    var = "b_a_fix"
    for g in range(len(glaciers)):
        gl = glaciers[g]  # One glacier
        height = {}
        for stake in glStakes_20years[gl]:
            # Get coordinates and time of file for this stake:
            fileName = re.split(".csv", stake)[0][:-3]
            df_stake = read_stake_csv(path_glacattr, stake,
                                    COI).sort_values(by="date_fix0")

            # remove category 0
            df_stake = df_stake[df_stake.vaw_id > 0]

            height[fileName] = df_stake.height.iloc[0]  # Height of stake

        # Sort stakes per elevation
        stakes_per_el[gl] = list(
            pd.Series(height).sort_values(ascending=True).index.values)

    rename_stakes = {}
    for gl in stakes_per_el.keys():
        for i, stake in enumerate(stakes_per_el[gl]):
            rename_stakes[stake] = f"{GLACIER_CORRECT[gl]}-P{i+1}"
    
    return rename_stakes