from pymatreader import read_mat
from matplotlib import pyplot as plt
import os
import sys
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import re
from random import sample
import pickle
from scipy.stats import pearsonr
import random as rd
import torch
from itertools import chain, combinations, permutations

from scipy.io import loadmat  # this is the SciPy module that loads mat-files
from datetime import datetime, date, time
from datetime import datetime

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
import joblib

from scripts.wgs84_ch1903 import *


# Converts .dat files to .csv
def processDatFile(fileName, path_dat, path_csv):
    with open(path_dat + fileName + '.dat', 'r') as dat_file:
        with open(path_csv + fileName + '.csv', 'w', newline='') as csv_file:
            num_rows = 0
            for row in dat_file:
                if num_rows == 1:
                    row = [value.strip() for value in row.split(';')]
                    csv_file.write(','.join(row) + '\n')
                if num_rows > 3:
                    row = [value.strip() for value in row.split(' ')]
                    # remove empty spaces
                    row = [i for i in row if i]
                    csv_file.write(','.join(row) + '\n')
                num_rows += 1


def datetime_obj(value):
    date  = str(value)
    year  = date[:4]
    month = date[4:6]
    day   = date[6:8]
    return pd.to_datetime(month + '-' + day + '-' + year)


def middleDates(value, year):
    date = str(value)
    if len(date) == 4:
        month = date[:2]
        day = date[2:]
    elif len(date) == 3:
        month = date[:1]
        day = date[1:]
    return month + '-' + day + '-' + str(year)


def transformDates(df_or):
    """Some dates are missing in the original glamos data and need to be corrected.
    Args:
        df_or (pd.DataFrame): raw glamos dataframe
    Returns:
        pd.DataFrame: dataframe with corrected dates
    """
    df = df_or.copy()
    # Correct dates that have years:
    df.date0 = df.date0.apply(lambda x: datetime_obj(x))
    df.date1 = df.date1.apply(lambda x: datetime_obj(x))

    df['date_fix0'] = [np.nan for i in range(len(df))]
    df['date_fix1'] = [np.nan for i in range(len(df))]

    # transform rest of date columns who have missing years:
    for i in range(len(df)):
        year = df.date0.iloc[i].year
        df.date_fmeas.iloc[i] = middleDates(df.date_fmeas.iloc[i], year)

        df.date_fmin.iloc[i]  = middleDates(df.date_fmin.iloc[i], year)

        df.date_smeas.iloc[i] = middleDates(df.date_smeas.iloc[i], year + 1)

        df.date_smax.iloc[i]  = middleDates(df.date_smax.iloc[i], year + 1)

        df.date_fix0.iloc[i] = '10' + '-' + '01' + '-' + str(year)
        df.date_fix1.iloc[i] = '09' + '-' + '30' + '-' + str(year + 1)

    df.date_fmeas = pd.to_datetime(df.date_fmeas)
    df.date_fmin  = pd.to_datetime(df.date_fmin)
    df.date_smeas = pd.to_datetime(df.date_smeas)
    df.date_smax  = pd.to_datetime(df.date_smax)
    df.date_fix0  = pd.to_datetime(df.date_fix0)
    df.date_fix1  = pd.to_datetime(df.date_fix1)
    return df


def LV03toWGS84(df):
    """Converts from swiss data coordinate system to lat/lon/height
    Args:
        df (pd.DataFrame): data in x/y swiss coordinates
    Returns:
        pd.DataFrame: data in lat/lon/coords
    """
    converter = GPSConverter()
    lat, lon, height = converter.LV03toWGS84(df['x'], df['y'], df['z'])
    df['lat'] = lat
    df['lon'] = lon
    df['height'] = height
    df.drop(['x', 'y', 'z'], axis=1, inplace=True)
    return df
