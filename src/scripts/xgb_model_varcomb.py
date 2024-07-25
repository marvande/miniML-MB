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
from scripts.xgb_model import *
from scripts.xgb_metrics import *
from scripts.xgb_input import *
from scripts.stakes_processing import *


def runXGBoost_varcomb(
        combinations_t2m_tp,
        hp_lr,
        hp_ne,
        hp_md,
        glStakes_20years,  # stakes and glaciers
        param_grid,  # grid for HP search
        path,
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
        log=False,
        empty_folder=False,
        tp_sum=True):
    if kfold:
        fold = "kfold"
    else:
        fold = "single_fold"

    vars_ = list(input_vars.keys())[0]
    for var in list(input_vars.keys())[1:]:
        vars_ += "_" + var

    # Create paths to save intermediate result:
    print(f"Creating path: {path}")

    # Empty folder and create path if non existance
    createPath(path)
    if empty_folder:
        emptyfolder(path)
        glRemaining = glStakes_20years
    else:
        # Check stakes already processed:
        allStakes = []
        for gl in glStakes_20years.keys():
            allStakes.append(glStakes_20years[gl])
        allStakes = np.concatenate(allStakes)
        allStakes = [
            re.split('_', f)[0] + '_' + re.split('_', f)[1] for f in allStakes
        ]
        stakes_processed = [
            re.split('_', f)[0] + '_' + re.split('_', f)[1][:-4]
            for f in os.listdir(path)
        ]
        remaining_stakes = Diff(list(stakes_processed), list(allStakes))
        print(allStakes)
        print('Already processed:\n', stakes_processed)
        print('Remaining stakes:\n', remaining_stakes)
        glRemaining = {}
        for stake in remaining_stakes:
            glacier = re.split('_', stake)[0]
            updateDic(glRemaining, glacier, stake + '_mb.csv')

    # Get list of glaciers
    glaciers = list(glRemaining.keys())

    print(f"Matching {mb_match} MB:\n------------------")

    # for gl in tqdm(glaciers, desc="glaciers"):
    for stakeName in tqdm(remaining_stakes, desc="stakes", leave=False):

        # Read MB data:
        #stakeName = re.split(".csv", stake)[0][:-3]
        df_stake = read_stake_csv(path_glacattr, stake + '_mb.csv')

        print(f"Running XGB for stake: {stakeName}\n-------------")

        if input_type == "ERA5-Land":
            # Read corresponding era 5 land values for this stake:
            xr_temppr = xr.open_dataset(
                path_era5_stakes + f"{stakeName}_mb_full.nc").sortby("time")

        if input_type == "MeteoSuisse":
            # Read corresponding meteo suisse values for this stake:
            xr_temppr = xr.open_dataset(
                path_MS + f"{stakeName}_mb_full.nc").sortby("time")

        begin_xr = pd.to_datetime(xr_temppr["time"].values[0]).year
        end_xr = pd.to_datetime(xr_temppr["time"].values[-1]).year

        # Cut MB data to same years as xr era 5:
        df_stake_cut = cutStake(df_stake, begin_xr, end_xr)

        # Remove cat 0 (only modelled MB)
        target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]

        test_rmse, val_rmse, train_rmse = [], [], []
        test_mae, val_mae, train_mae = [], [], []
        dfcombi = pd.DataFrame(combinations_t2m_tp, columns=['t2m', 'tp'])
        for combN in tqdm(range(len(combinations_t2m_tp)),
                          desc="combinations",
                          leave=False):
            combination_t2m = combinations_t2m_tp[combN][0]
            combination_tp = combinations_t2m_tp[combN][1]
            # Create input data for that combi of t2m var:
            inputDF = createInputDF_varcomb(combination_t2m,
                                            combination_tp,
                                            df_stake_cut,
                                            xr_temppr,
                                            month_val=month_val,
                                            input_type=input_type,
                                            match="annual",
                                            tp_sum=tp_sum,
                                            vars_=list(input_vars.keys()))
            inputDF_winter = createInputDF_varcomb(combination_t2m,
                                                   combination_tp,
                                                   df_stake_cut,
                                                   xr_temppr,
                                                   month_val=month_val,
                                                   input_type=input_type,
                                                   match="annual",
                                                   tp_sum=tp_sum,
                                                   vars_=list(
                                                       input_vars.keys()))
            input_ = {"annual": inputDF, "winter": inputDF_winter}

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

            if log:
                print('Input DF shape: {}'.format(input_['annual'].shape))
                print(f'Target shape: {target.shape}')

            # Apply XGBoost model
            custom_params = {
                'learning_rate': hp_lr[stakeName],
                'n_estimators': int(hp_ne[stakeName]),
                'max_depth': int(hp_md[stakeName])
            }
            (predictions_XG, feature_import, fi_all, feat_test, feat_train,
             eval_metrics, fold_ids, validation,
             HP_RF) = applyXGBoost(input_,
                                   target,
                                   add_2022,
                                   kfold,
                                   mb_match,
                                   param_grid,
                                   log=log,
                                   grid_search=False,
                                   custom_params_RF=custom_params)
            test_rmse.append(np.mean(eval_metrics["rmse"]))
            test_mae.append(np.mean(eval_metrics["mae"]))

            # save last validation loss over all folds
            val_rmse.append(
                np.mean([
                    validation["val"][i]['rmse'][-1]
                    for i in range(len(validation))
                ]))

            val_mae.append(
                np.mean([
                    validation["val"][i]['mae'][-1]
                    for i in range(len(validation))
                ]))

            train_rmse.append(
                np.mean([
                    validation["train"][i]['rmse'][-1]
                    for i in range(len(validation))
                ]))

            train_mae.append(
                np.mean([
                    validation["train"][i]['mae'][-1]
                    for i in range(len(validation))
                ]))

        dfcombi['test_rmse'] = test_rmse
        dfcombi['val_rmse'] = val_rmse
        dfcombi['train_rmse'] = train_rmse
        dfcombi['test_mae'] = test_mae
        dfcombi['val_mae'] = val_mae
        dfcombi['train_mae'] = train_mae

        nameDf = f"{stakeName}.csv"
        dfcombi.to_csv(path + nameDf,
                       columns=[
                           't2m', 'tp', 'test_rmse', 'val_rmse', 'train_rmse',
                           'test_mae', 'val_mae', 'train_mae'
                       ])
        print(f"DF {nameDf} saved successfully to {path}")


def runXGBoost_one_varcomb(
        combination,
        hp_lr,
        hp_ne,
        hp_md,
        glStakes_20years,  # stakes and glaciers
        param_grid,  # grid for HP search
        weights_t2m,
        weights_tp,
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
        log=False,
        empty_folder=False,
        grid_search=False):
    if kfold:
        fold = "kfold"
    else:
        fold = "single_fold"

    vars_ = list(input_vars.keys())[0]
    for var in list(input_vars.keys())[1:]:
        vars_ += "_" + var

    # Create paths to save intermediate result:
    variables = list(input_vars.keys())[0] + '_' + list(input_vars.keys())[1]
    if np.sum(weights_t2m) != len(weights_t2m):
        variables += "_weighted"    

    path = path_save_xgboost_stakes + f"{fold}/{input_type}/best_combi/grid_search_{grid_search}/{variables}/{SEED}/"
    print(f"Creating path: {path}")

    # Empty folder and create path if non existance
    createPath(path)
    if empty_folder:
        emptyfolder(path)
        glRemaining = glStakes_20years
    else:
        # Check stakes already processed:
        allStakes = []
        for gl in glStakes_20years.keys():
            allStakes.append(glStakes_20years[gl])
        allStakes = np.concatenate(allStakes)
        allStakes = [
            re.split('_', f)[0] + '_' + re.split('_', f)[1] for f in allStakes
        ]

        allFiles = [
            re.split("var_", f)[1][:-4] for f in os.listdir(path)
            if (f[:3] == "var")
        ]
        stakes_processed = np.unique([
            re.split('_', f)[0] + '_' + re.split('_', f)[1] for f in allFiles
        ])
        remaining_stakes = Diff(list(stakes_processed), list(allStakes))
        glRemaining = {}
        for stake in remaining_stakes:
            glacier = re.split('_', stake)[0]
            updateDic(glRemaining, glacier, stake + '_mb.csv')

    # Get list of glaciers
    glaciers = list(glRemaining.keys())

    print(f"Matching {mb_match} MB:\n------------------")

    for gl in tqdm(glaciers, desc="glaciers"):
        for stakeNb in range(len(glRemaining[gl])):

            # Read MB data:
            stake = glRemaining[gl][stakeNb]
            stakeName = re.split(".csv", stake)[0][:-3]
            df_stake = read_stake_csv(path_glacattr, stake)

            # Get best combination
            # print(f"Running XGB for stake: {stakeName}\n-------------")

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

            # Cut MB data to same years as xr era 5:
            df_stake_cut = cutStake(df_stake, begin_xr, end_xr)

            # Remove cat 0 (only modelled MB)
            target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]

            # Create input data for that combi of t2m var:
            combination_t2m = combination[0][0]
            combination_tp = combination[0][1]
            inputDF = createInputDF_varcomb(combination_t2m,
                                            combination_tp,
                                            df_stake_cut,
                                            xr_temppr,
                                            weights_t2m,
                                            weights_tp,
                                            month_val=month_val,
                                            input_type=input_type,
                                            match="annual",
                                            tp_sum=True,
                                            vars_=list(input_vars.keys()))
            inputDF_winter = createInputDF_varcomb(combination_t2m,
                                                   combination_tp,
                                                   df_stake_cut,
                                                   xr_temppr,
                                                   weights_t2m,
                                                   weights_tp,
                                                   month_val=month_val,
                                                   input_type=input_type,
                                                   match="annual",
                                                   tp_sum=True,
                                                   vars_=list(
                                                       input_vars.keys()))
            input_ = {"annual": inputDF, "winter": inputDF_winter}

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

            if log:
                print('Input DF shape: {}'.format(input_['annual'].shape))
                print(f'Target shape: {target.shape}')

            # Apply XGBoost model
            custom_params = {
                'learning_rate': hp_lr[stakeName],
                'n_estimators': int(hp_ne[stakeName]),
                'max_depth': int(hp_md[stakeName])
            }
            if log:
                print('Running model with grid search: ', grid_search)
            (predictions_XG, feature_import, fi_all, feat_test, feat_train,
             eval_metrics, fold_ids, validation,
             HP_RF) = applyXGBoost(input_,
                                   target,
                                   add_2022,
                                   kfold,
                                   mb_match,
                                   param_grid,
                                   log=log,
                                   grid_search=grid_search,
                                   custom_params_RF=custom_params)
            if log:
                print('Best parameters:', HP_RF)

            # Save variables in pkl file
            var = {
                "rmse": eval_metrics["rmse"],
                "mae": eval_metrics["mae"],
                "correlation": eval_metrics["correlation"],
                "rsquared": eval_metrics["rsquared"],
                "variables": combination,
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
                if log:
                    print(f"dictionary {name} saved successfully to {path}")


def createPrec_winter_halfyear(Pdf, Ptempprxr):
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

        xr_temppr_why = Ptempprxr.sel(time=winter_range)

        # half year sum and average
        yearDF_why_prec = xr_temppr_why.tp.sum().item(0)

        vars_[year + 1] = yearDF_why_prec

    inputFeatures = pd.DataFrame(data={
        'year': vars_.keys(),
        'prec_WHY': vars_.values()
    }).set_index('year')

    return inputFeatures


def createInputDF_varcomb(combination_t2m,
                          combination_prec,
                          df_stake_cut,
                          xr_temppr,
                          weights_t2m,
                          weights_tp,
                          vars_,
                          tp_sum=True,
                          month_val=MONTH_VAL,
                          input_type='MeteoSuisse',
                          match="annual",
                          unseen = False):

    # Create monthly DF for t2m
    inputDF = createInputDF(df_stake_cut,
                            xr_temppr,
                            month_val=month_val,
                            long_vars={vars_[0]: 'temperature'},
                            input_type=input_type,
                            match=match,
                            unseen = unseen)
    # inputDF_mean_t2m = pd.DataFrame(
    #     inputDF[combination_t2m].mean(axis=1)).rename(columns={0: 't2m_mean'})
    # weighted mean (if no weights provided then weights are 1)
    weightsDF_ = inputDF[combination_t2m].transpose()
    weightsDF_['weights'] = weights_t2m
    weighted_means, index = [], []
    for col in weightsDF_.columns:
        if col != 'weights':
            weighted_means.append(np.average(weightsDF_[col].values, weights = weightsDF_['weights'].values))
            index.append(col)
    inputDF_mean_t2m = pd.DataFrame(index = index, data = weighted_means, columns = ['t2m_mean'])        
        
    # Create monthly DF for prec
    inputDF = createInputDF(df_stake_cut,
                            xr_temppr,
                            month_val=month_val,
                            long_vars={vars_[1]: 'total precipitation'},
                            input_type=input_type,
                            match=match,
                            unseen = unseen)
    # total prec
    if tp_sum:
        # inputDF_mean_prec = pd.DataFrame(inputDF[combination_prec].sum(
        #     axis=1)).rename(columns={0: 'tp_tot'})
        
        # weighted sum (if no weights provided then weights are 1)
        weightsDF_ = inputDF[combination_prec].transpose()
        weightsDF_['weights'] = weights_tp
        weighted_sum, index = [], []
        for col in weightsDF_.columns:
            if col != 'weights':
                weighted_sum.append(np.dot(weightsDF_[col].values, weightsDF_['weights'].values))
                index.append(col)
        inputDF_mean_prec = pd.DataFrame(index = index, data = weighted_sum, columns = ['tp_tot'])    
    else:
        # inputDF_mean_prec = pd.DataFrame(inputDF[combination_prec].mean(
        #     axis=1)).rename(columns={0: 'tp_tot'})
        
        # weighted mean (if no weights provided then weights are 1)
        weightsDF_ = inputDF[combination_prec].transpose()
        weightsDF_['weights'] = weights_tp
        weighted_means, index = [], []
        for col in weightsDF_.columns:
            if col != 'weights':
                weighted_means.append(np.average(weightsDF_[col].values, weights = weightsDF_['weights'].values))
                index.append(col)
        inputDF_mean_prec = pd.DataFrame(index = index, data = weighted_means, columns = ['tp_tot'])        
        

    # Combinate temp mean and precipitation winter half year
    inputDF_combi = pd.concat([inputDF_mean_t2m, inputDF_mean_prec], axis=1)
    inputDF_combi.sort_index(inplace=True)
    return inputDF_combi


def makeCombNum(combination1, combination2):
    t2m_primes = {
        't2m_Oct': 2,
        't2m_Nov': 3,
        't2m_Dec': 5,
        't2m_Jan': 7,
        't2m_Feb': 11,
        't2m_Mar': 13,
        't2m_Apr': 17,
        't2m_May': 19,
        't2m_June': 23,
        't2m_July': 31,
        't2m_Aug': 37,
        't2m_Sep': 41,
    }
    tp_primes = {
        'tp_Oct': 43,
        'tp_Nov': 47,
        'tp_Dec': 53,
        'tp_Jan': 59,
        'tp_Feb': 61,
        'tp_Mar': 67,
        'tp_Apr': 71,
        'tp_May': 73,
        'tp_June': 79,
        'tp_July': 83,
        'tp_Aug': 89,
        'tp_Sep': 97,
    }
    mult = 1
    for var_month in combination1:
        mult *= t2m_primes[var_month]
    for var_month in combination2:
        mult *= tp_primes[var_month]
    return mult


def NBestCombinations(dfAllStakes,
                      month_pos,
                      t2m_vars,
                      tp_vars,
                      N=50,
                      type='val_rmse'):
    """
    Calculate the N best combinations of features and their weights for each stake.

    Args:
        dfAllStakes (DataFrame): DataFrame containing stakes and their associated features.
        month_pos (dict): Dictionary mapping month names to their corresponding positions.
        t2m_vars (list): List of t2m features.
        tp_vars (list): List of tp features.
        N (int, optional): Number of best combinations to consider. Defaults to 50.

    Returns:
        tuple: A tuple containing two DataFrames:
            - feature_importdf: DataFrame containing the feature importances for each combination.
            - dfWeights: DataFrame containing the weights for each feature in each combination.
    """
    t2m_weight_stakes, tp_weight_stakes, t2m_feat, tp_feat, stakes = [], [], [], [], []
    val_rmse = []
    for stake in dfAllStakes.stakes.unique():
        N_best = dfAllStakes[dfAllStakes.stakes == stake].sort_values(
            by=type)[:N]
        tp_weight_map = np.zeros(12)
        t2m_weight_map = np.zeros(12)

        count_t2m = N_best.explode('t2m').groupby('t2m').count().reset_index()
        count_tp = N_best.explode('tp').groupby('tp').count().reset_index()

        for i, var in enumerate(count_tp['tp']):
            month = re.split('_', var)[1]
            tp_weight_map[month_pos[month]] = count_tp.iloc[i]['glaciers']

        for i, var in enumerate(count_t2m['t2m']):
            month = re.split('_', var)[1]
            t2m_weight_map[month_pos[month]] = count_t2m.iloc[i]['glaciers']

        t2m_feat.append(t2m_vars)
        tp_feat.append(tp_vars)
        t2m_weight_stakes.append(t2m_weight_map)
        tp_weight_stakes.append(tp_weight_map)
        stakes.append(np.tile(stake, 12))
        val_rmse.append(np.tile(np.mean(N_best[type]), 12))

    feature_importdf = pd.DataFrame({
        't2m_feat':
        np.concatenate(t2m_feat),
        'tp_feat':
        np.concatenate(tp_feat),
        'stakes':
        np.concatenate(stakes),
        't2m_weight':
        np.concatenate(t2m_weight_stakes),
        'tp_weight':
        np.concatenate(tp_weight_stakes),
        type:
        np.concatenate(val_rmse)
    })
    feature_importdf['month'] = feature_importdf.t2m_feat.apply(
        lambda x: re.split('_', x)[1])

    dfWeights = pd.DataFrame({
        'weight':
        pd.concat(
            [feature_importdf['t2m_weight'], feature_importdf['tp_weight']],
            axis=0),
        type:
        pd.concat([feature_importdf[type], feature_importdf[type]], axis=0),
        'stakes':
        pd.concat([feature_importdf['stakes'], feature_importdf['stakes']],
                  axis=0),
        'month':
        pd.Categorical(pd.concat(
            [feature_importdf['month'], feature_importdf['month']], axis=0),
                       ordered=True),
        'feature':
        np.concatenate([
            np.tile('t2m', len(feature_importdf['stakes'])),
            np.tile('tp', len(feature_importdf['stakes']))
        ])
    })

    return feature_importdf, dfWeights


def NBestCombinations_avgStakes(dfAllStakes,
                                month_pos,
                                t2m_vars,
                                tp_vars,
                                N=50,
                                type='val_rmse'):
    """
    Calculate the N best combinations of features over all stakes.

    Args:
        dfAllStakes (DataFrame): The DataFrame containing all the stakes.
        month_pos (dict): A dictionary mapping month names to their positions.
        t2m_vars (list): A list of t2m variables.
        tp_vars (list): A list of tp variables.
        N (int, optional): The number of best combinations to consider. Defaults to 50.

    Returns:
        tuple: A tuple containing two DataFrames:
            - feature_importdf: DataFrame with feature importance information.
            - dfWeights: DataFrame with weights, test RMSE, month, and feature information.
    """
    # Average over all stakes (instead of ind stakes as before)
    avgAllStakes = dfAllStakes.groupby('t2m-tp-hash').mean().sort_values(
        by=type)

    N_best = avgAllStakes[:N].reset_index()
    # Add back the lists of combinations
    N_best['t2m'] = [
        dfAllStakes[dfAllStakes['t2m-tp-hash'] == N_best['t2m-tp-hash']
                    [i]].iloc[0].t2m for i in range(len(N_best))
    ]
    N_best['tp'] = [
        dfAllStakes[dfAllStakes['t2m-tp-hash'] == N_best['t2m-tp-hash']
                    [i]].iloc[0].tp for i in range(len(N_best))
    ]

    # Look at feature importance of months over all best combinations
    tp_weight_map = np.zeros(12)
    t2m_weight_map = np.zeros(12)

    count_t2m = N_best.explode('t2m').groupby('t2m').count().reset_index()
    count_tp = N_best.explode('tp').groupby('tp').count().reset_index()

    for i, var in enumerate(count_tp['tp']):
        month = re.split('_', var)[1]
        tp_weight_map[month_pos[month]] = count_tp.iloc[i]['t2m-tp-hash']

    for i, var in enumerate(count_t2m['t2m']):
        month = re.split('_', var)[1]
        t2m_weight_map[month_pos[month]] = count_t2m.iloc[i]['t2m-tp-hash']

    feature_importdf = pd.DataFrame({
        't2m_feat': t2m_vars,
        'tp_feat': tp_vars,
        't2m_weight': t2m_weight_map,
        'tp_weight': tp_weight_map,
        type: np.tile(np.mean(N_best[type]), 12)
    })
    feature_importdf['month'] = feature_importdf.t2m_feat.apply(
        lambda x: re.split('_', x)[1])

    dfWeights = pd.DataFrame({
        'weight':
        pd.concat(
            [feature_importdf['t2m_weight'], feature_importdf['tp_weight']],
            axis=0),
        type:
        pd.concat([feature_importdf[type], feature_importdf[type]], axis=0),
        'month':
        pd.Categorical(pd.concat(
            [feature_importdf['month'], feature_importdf['month']], axis=0),
                       ordered=True),
        'feature':
        np.concatenate([
            np.tile('t2m', len(feature_importdf['month'])),
            np.tile('tp', len(feature_importdf['month']))
        ])
    })
    return feature_importdf, dfWeights


def BestCombination(best_comb_t2m_tp,
                    dfAllStakes,
                    month_pos,
                    t2m_vars,
                    tp_vars,
                    type='val_rmse'):
    rmse_comb = {}
    for comb in best_comb_t2m_tp:
        hash = makeCombNum(comb[0], comb[1])
        rmse_comb[hash] = dfAllStakes[dfAllStakes['t2m-tp-hash'] == hash][
            type].mean()  # mean over all stakes
    df = pd.DataFrame(data={
        't2m-tp-hash': rmse_comb.keys(),
        type: rmse_comb.values()
    })
    df['t2m'] = [comb[0] for comb in best_comb_t2m_tp]
    df['tp'] = [comb[1] for comb in best_comb_t2m_tp]

    # Look at feature importance of months over all best combinations
    tp_weight_map = np.zeros(12)
    t2m_weight_map = np.zeros(12)

    count_t2m = df.explode('t2m').groupby('t2m').count().reset_index()
    count_tp = df.explode('tp').groupby('tp').count().reset_index()

    for i, var in enumerate(count_tp['tp']):
        month = re.split('_', var)[1]
        tp_weight_map[month_pos[month]] = count_tp.iloc[i]['t2m-tp-hash']

    for i, var in enumerate(count_t2m['t2m']):
        month = re.split('_', var)[1]
        t2m_weight_map[month_pos[month]] = count_t2m.iloc[i]['t2m-tp-hash']

    feature_importdf = pd.DataFrame({
        't2m_feat': t2m_vars,
        'tp_feat': tp_vars,
        't2m_weight': t2m_weight_map,
        'tp_weight': tp_weight_map,
        type: np.tile(np.mean(df[type]), 12)
    })
    feature_importdf['month'] = feature_importdf.t2m_feat.apply(
        lambda x: re.split('_', x)[1])

    dfWeights = pd.DataFrame({
        'weight':
        pd.concat(
            [feature_importdf['t2m_weight'], feature_importdf['tp_weight']],
            axis=0),
        type:
        pd.concat([feature_importdf[type], feature_importdf[type]], axis=0),
        'month':
        pd.Categorical(pd.concat(
            [feature_importdf['month'], feature_importdf['month']], axis=0),
                       ordered=True),
        'feature':
        np.concatenate([
            np.tile('t2m', len(feature_importdf['month'])),
            np.tile('tp', len(feature_importdf['month']))
        ])
    })

    dfWeights['freq_var'] = dfWeights['weight'] / len(df)

    return feature_importdf, dfWeights, df


# Get all consecutive combinations of length max 6:
def consecutive_combinations(iterable, consec):
    begin = 0
    chunks = len(iterable) + 1 - consec
    return [iterable[x + begin:x + consec] for x in range(chunks)]
