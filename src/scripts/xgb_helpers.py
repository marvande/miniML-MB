import os
from os import listdir
from os.path import isfile, join
import numpy as np
import re
import random as rd
import torch
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from datetime import datetime

# --------------------------------------------------
# PATHS:
# Path to point mass balance
path_index_raw = '../../data/MB_modeling/GLAMOS/index/dat_files/'
mb_path = "../../data/MB_modeling/GLAMOS/index/csv_files/massbalance/"
path_latloncoord = mb_path + "WGSlatloncoord/"  # lat-lon coord
path_GLAMOS_csv = mb_path + "raw_csv/"  # raw glamos data
path_glacattr = mb_path + "glacierattr/"  # lat-lon coord + gl attributes from oggm

# Path to ERA5-land
path_ERA5 = "../../data/MB_modeling/ERA5/"
path_era5_stakes = (
    path_ERA5 + "ERA5Land-stakes/"
)  # path to era land at stakes coordinates
path_glogem = "../../data/GloGEM/dataframes/"
path_ERA5_Land = path_ERA5 + "ERA5-Land/"  # whole ERA5-land
path_ERA5_Land_hourly = path_ERA5 + "/ERA5-Land-hourly/ncfiles/"
path_GLAMOS = '../../data/MB_modeling/GLAMOS/'

# Path XGBoost and PDD model
path_pickles = "../../data/MB_modeling/PDD/"
path_save_xgboost = "../../data/MB_modeling/XGBoost/"
path_save_xgboost_stakes = "../../data/MB_modeling/XGBoost/ind_stakes/"

# Path linear model
path_save_LM = "../../data/MB_modeling/LinearModel/"
path_save_LM_stakes = "../../data/MB_modeling/LinearModel/ind_stakes/"

# Path meteo suisse: 
path_MS = '../../data/MB_modeling/MeteoSuisse/stakes/'
path_MS_full = '../../data/MB_modeling/MeteoSuisse/stakes_full/'
path_meteogrid = '../../data/MB_modeling/MeteoSuisse/'
path_prec = path_meteogrid+'RhiresM_verified/lonlat/'
path_temp = path_meteogrid+'/TabsM_verified/lonlat/'


# --------------------------------------------------
# Constants:
SEED = 5

# year goes from oct - sept
MONTH_VAL = {
    1: "Oct",
    2: "Nov",
    3: "Dec",
    4: "Jan",
    5: "Feb",
    6: "Mar",
    7: "Apr",
    8: "May",
    9: "June",
    10: "July",
    11: "Aug",
    12: "Sep",
}

INVERSE_MONTH_VAL = {
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "Aug": 8,
    "Sep": 9,
}

INVERSE_MONTH_POS = {
    "Oct": 0,
    "Nov": 1,
    "Dec": 2,
    "Jan": 3,
    "Feb": 4,
    "Mar": 5,
    "Apr": 6,
    "May": 7,
    "June": 8,
    "July": 9,
    "Aug": 10,
    "Sep": 11,
}

GL_SHORT = {
    "Basodino": "BAS",
    "Gries": "GRI",
    "Schwarzberg": "SCH",
    "Aletsch": "ALE",
    "Limmern": "LIM",
    "Clariden": "CLA",
    "Allalin": "ALL",
    "Silvretta": "SIL",
    "Hohlaub": "HOH",
    "Pers": "PERS",
    "Corbassiere": "COR",
    "Plattalva": "PLA",
    "Gietro": "GIE",
}

# columns of interest in glamos data:
COI = [
    "glims_id",
    "sgi_id",
    "rgi_id",
    "glims_id",
    "vaw_id",
    "date_fix0",
    "date_fix1",
    "date0",
    "date1",
    "date_smeas",
    "lat",
    "lon",
    "height",
    "b_a_fix",
    "b_w_fix",
    "aspect",
    "slope",
    "dis_from_border",
    "min_el_gl",
    "max_el_gl",
    "med_el_gl",
]

LONG_VARS = {
    "t2m": "temperature",
    "tp": "precipitation",
    "sd": "snow-depth",
    "sde": "snow-depth-2",
    "u10": "U-wind",
    "v10": "V-wind",
    "sp": "surface-pressure",
    "fal": "albedo",
    "slhf": "surface-latent-heat-flux",
    "ssrd": "surface-solar-rad-down",
    "sshf": "surface-sensible-heat-flux",
    "strd": "surface-thermal-rad-down",
    "ssr": "surface-net-solar-rad",
    "str": "surface-net-rad",
}

GLACIER_CORRECT = {
    'aletsch': 'Aletsch',
    'allalin': 'Allalin',
    'basodino': 'Basodino',
    'clariden': 'Clariden',
    'corbassiere': 'Corbassiere',
    'gietro': 'Gietro',
    'gries': 'Gries',
    'hohlaub': 'Hohlaub',
    'limmern': 'Limmern',
    'pers': 'Pers',
    'plattalva': 'Plattalva',
    'schwarzberg': 'Schwarzberg',
    'silvretta': 'Silvretta'
}

def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_long_name_metric(name):
    splits = re.split("_", name)
    long_names_metrics = {
        "diff_rmse": "Difference in RMSE to TIM",
        "diff_mae": "Difference in MAE to TIM",
        "diff_corr": "Difference in correlation to TIM",
        "diff_r2": "Difference in $\mathrm{R^2}$ to TIM",
        "mae": "Mean absolute error",
        "rmse": "RMSE",
        "nrmse": "Normalised root mean squared error",
        "corr": "Pearson correlation coefficient",
        "r2": "$\mathrm{R}^2$",
    }

    y_labels = {
        "mae": "[m w.e. y$^{-1}$]",
        "rmse": "[m w.e. y$^{-1}$]",
        "corr": "",
        "r2": "",
        "nrmse": "",
        "diff_rmse": "[m w.e. y$^{-1}$]",
        "diff_mae": "[m w.e. y$^{-1}$]",
        "diff_corr": "",
        "diff_r2": "",
    }
    if splits[0] == "diff":
        long_name = long_names_metrics[splits[0] + "_" + splits[1]]
        ylabel = y_labels[splits[0] + "_" + splits[1]]
    else:
        long_name = long_names_metrics[splits[0]]
        ylabel = y_labels[splits[0]]

    return long_name, ylabel


def makeStr(month):
    if month / 10 < 1:
        str_month = "0" + str(month)
    else:
        str_month = str(month)
    return str_month


def createPath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# empties a folder
def emptyfolder(path):
    if os.path.exists(path):
        onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
        for f in onlyfiles:
            os.remove(path + f)
    else:
        createPath(path)


# difference between two lists
def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i not in li1 or i not in li2]
    return li_dif


# sets the same random seed everywhere so that it is reproducible
def seed_all(seed):
    if not seed:
        seed = 10
        # print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    rd.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Updates a dictionnary at key with value
def updateDic(dic, key, value):
    if key not in dic.keys():
        dic[key] = [value]
    else:
        dic[key].append(value)

    return dic


# Initialises dictionnaries for all glaciers
def initialiseDic(glaciers):
    dic = {}
    for gl in glaciers:
        dic[gl] = []
    return dic


# gets the number of stakes in a glacier lists
def get_NumPred(stake_list, pred_gl):
    num_pred = 0
    for key in stake_list:
        num_pred += len(pred_gl[key])
    return num_pred


def findOverlapPeriod(stake_year0, stake_year1, era5_year0, era5_year1):
    if stake_year0 < era5_year0:
        begin = era5_year0
    elif stake_year0 >= era5_year0:
        begin = stake_year0
    if era5_year1 < stake_year1:
        end = era5_year1
    elif era5_year1 >= stake_year1:
        end = stake_year1
    return (begin, end)


def remAdd2022(add_2022, isthere, inputDF, test_index, year = 2022):
    # if we want 2022 to be in the test data
    if isthere and add_2022:
        pos2022 = inputDF.index.get_loc(year)
        if pos2022 not in test_index:
            test_index = np.append(test_index, pos2022)
    # if we want to remove 2022
    if isthere and not add_2022:
        pos2022 = inputDF.index.get_loc(year)
        if pos2022 in test_index:
            test_index = list(test_index)
            test_index.remove(pos2022)
            test_index = np.array(test_index)
    return test_index


def remAdd2021(add_2022, isthere, inputDF, test_index):
    # if we want 2022 to be in the test data
    if isthere and add_2022:
        pos2022 = inputDF.index.get_loc(2021)
        if pos2022 not in test_index:
            test_index = np.append(test_index, pos2022)

    # if we want to remove 2022
    if isthere and not add_2022:
        pos2022 = inputDF.index.get_loc(2021)
        if pos2022 in test_index:
            test_index = list(test_index)
            test_index.remove(pos2022)
            test_index = np.array(test_index)

    return test_index


def getXYTime(inputDF, target, index, type_target="b_a_fix"):
    # Create test and training features
    X = inputDF.iloc[index].values
    y = target[type_target].iloc[index].values
    time = inputDF.iloc[index].index

    return X, y, time


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def assemblePredDf(time_test, pred, fold_ids, truth_test, time_train,
                   truth_train, kfold):
    """
    Assembles a pandas DataFrame containing prediction results.

    Args:
        time_test (array-like): Array of timestamps for test data.
        pred (array-like): Array of predicted values.
        fold_ids (array-like): Array of fold IDs.
        truth_test (array-like): Array of true values for test data.
        time_train (array-like): Array of timestamps for training data.
        truth_train (array-like): Array of true values for training data.
        kfold (bool): Flag indicating whether k-fold cross-validation was used.

    Returns:
        pandas.DataFrame: DataFrame containing the assembled prediction results.
    """
    df_test = pd.DataFrame(
        data={
            "time": time_test,
            "pred": pred,
            "truth": truth_test,
            "error": pred - truth_test,
            "fold_ids": fold_ids,
        })

    if kfold != True:
        df_train = pd.DataFrame(
            data={
                "time": time_train,
                "pred": [np.nan for i in range(len(time_train))],
                "truth": truth_train,
                "error": [np.nan for i in range(len(time_train))],
                "fold_ids": np.tile(fold_ids[0], len(time_train)),
            })

        total_df = pd.concat([df_test, df_train]).sort_values(by="time")

    else:
        total_df = df_test.sort_values(by="time")

    return total_df

def get_cmap_hex(cmap, length):
    """
    Function to get a get a list of colours as hex codes

    :param cmap:    name of colourmap
    :type cmap:     str

    :return:        list of hex codes
    :rtype:         list
    """
    # Get cmap
    rgb = plt.get_cmap(cmap)(np.linspace(0, 1, length))

    # Convert to hex
    hex_codes = [to_hex(rgb[i,:]) for i in range(rgb.shape[0])]

    return hex_codes

def days_between(d1, d2):
    d1 = datetime.strptime(d1, "%Y-%m-%d")
    d2 = datetime.strptime(d2, "%Y-%m-%d")
    return abs((d2 - d1).days)