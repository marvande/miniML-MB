import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm

from scripts.xgb_helpers import *
from scripts.xgb_input import *


def minMaxScaler(var_xg, array):
    target_vals = []
    for stake in var_xg["feat_test"].keys():
        target_vals.append(var_xg["feat_test"][stake]["target"])

    target_vals = np.concatenate(target_vals).reshape(-1, 1)

    # fit scaler:
    scaler = MinMaxScaler()
    scaler.fit(target_vals)

    # transform other array
    return scaler.transform(array)


def get_target_stats(feat, type_="target", type_pred="pred_XG"):
    if type_pred == "winter_pred_PDD":
        suffix = "_w"

    elif type_pred == "annual_pred_PDD":
        suffix = "_a"
    else:
        suffix = ""

    if isinstance(feat, dict):
        target_vals = []
        for stake in feat.keys():
            if type_ == "target":
                target_vals.append(feat[stake]["target" + suffix])
            else:
                target_vals.append(feat[stake])
        target_vals = np.concatenate(target_vals)

    else:
        target_vals = feat

    stats = {
        "lower_quantile": np.quantile(target_vals, q=0.25, axis=0),
        "upper_quantile": np.quantile(target_vals, q=0.75, axis=0),
        "mean": np.mean(target_vals),
        "median": np.median(target_vals),
        "min": np.min(target_vals),
        "max": np.max(target_vals),
    }

    stats["iqr"] = stats["upper_quantile"] - stats["lower_quantile"]

    stats["min_"] = stats["lower_quantile"] - 1.5 * stats["iqr"]
    stats["max_"] = stats["upper_quantile"] + 1.5 * stats["iqr"]

    return stats


def r_squared(truth, pred):
    corr_matrix = np.corrcoef(truth, pred)
    corr = corr_matrix[0, 1]
    R_sq = corr**2
    return R_sq


def r_squared_adj(truth, pred, num_feat):
    # R2 = r2_score(truth, pred)
    R2 = r_squared(truth, pred)
    rsquared_adj = 1 - (1 - R2) * (len(truth) - 1) / (len(truth) - num_feat -
                                                      1)
    return rsquared_adj


def evalMetrics(truth, pred):
    rmse = mean_squared_error(pred, truth, squared=False)
    mae = mean_absolute_error(truth, pred)

    if len(pred) >= 2:
        pearson = pearsonr(pred, truth)[0]
    else:
        pearson = np.nan
    #rsquared = r2_score(pred, truth)
    rsquared2 = r_squared(truth, pred)

    return rmse, mae, pearson, rsquared2


def dfMetricsCorrected(
    var_xg,
    kfold,
    type_pred="pred_XG",
):
    if type_pred == "winter_pred_PDD":
        suffix = "_w"
        suffix_name = "_pdd_w"
    elif type_pred == "annual_pred_PDD":
        suffix = "_a"
        suffix_name = "_pdd_a"
    elif type_pred == 'pred_XG':
        suffix = ""
        suffix_name = "_xgb"
    elif type_pred == 'RF':
        suffix = ""
        suffix_name = "_rf"
    elif type_pred == 'pred_LM':
        suffix = ""
        suffix_name = "_lasso"
    gl_names, stakes, stakes_full = [], [], []
    rmse, corr, mae, rsquared, num_points = [], [], [], [], []
    std_obs = []
    for stakeName in var_xg[type_pred].keys():
        # Construct whole time series over all hydrological years:
        dfStake = pd.DataFrame({
            type_pred:
            var_xg[type_pred][stakeName],
            'target':
            var_xg['feat_test'][stakeName]['target' + suffix],
            'time':
            var_xg['feat_test'][stakeName]['time']
        })
        dfStake.sort_values(by='time', inplace=True)

        # Info:
        gl_names.append(stakeName.split("-")[0])
        stakes_full.append(GL_SHORT[stakeName.split("-")[0]] + "-" +
                           stakeName.split("-")[1])
        stakes.append(stakeName.split("-")[1])
        num_points.append(len(var_xg['feat_test'][stakeName]['time']) / kfold)
        # Metrics:
        target = dfStake['target'] / (1000)  # m w.e.
        pred = dfStake[type_pred] / (1000)  # m w.e.
        rmse.append(mean_squared_error(pred, target, squared=False))
        corr.append(pearsonr(pred, target)[0])
        mae.append(mean_absolute_error(pred, target))

        rsquared.append(r_squared(pred, target))

        # std obs:
        std_obs.append(np.std(target))

    data = {
        "glaciers": gl_names,
        "stakes": stakes,
        "stakes_full": stakes_full,
        "rmse" + suffix_name: rmse,
        "corr" + suffix_name: corr,
        "mae" + suffix_name: mae,
        "r2" + suffix_name: rsquared,
        "num_points": num_points,
        "std_obs": std_obs
    }
    df_metrics = pd.DataFrame(data)

    return df_metrics


def dfMetrics(
    pred_XG_stake,
    metrics,
    kfold,
    type_pred="pred_XG",
):
    if type_pred == "winter_pred_PDD":
        suffix = "_w"
        suffix_name = "_pdd_w"
    elif type_pred == "annual_pred_PDD":
        suffix = "_a"
        suffix_name = "_pdd_a"
    elif type_pred == 'pred_XG':
        suffix = ""
        suffix_name = "_xgb"
    elif type_pred == 'RF':
        suffix = ""
        suffix_name = "_rf"
    elif type_pred == 'pred_LM':
        suffix = ""
        suffix_name = "_lasso"

    # metrics per stake
    rmse_stake = metrics["rmse" + suffix]
    if type_pred == 'pred_XG':
        rmse_stake_val = metrics["val_loss"]
        rmse_stake_train = metrics["train_loss"]
    elif type_pred == 'RF' or type_pred == 'pred_LM':
        rmse_stake_val = metrics["rmse_val"]
        rmse_stake_train = metrics["rmse_train"]

    mae_stake = metrics["mae" + suffix]
    pearson_stake = metrics["correlation" + suffix]
    rsquared_stake = metrics["rsquared" + suffix]

    gl_names, stakes, stakes_full, rmse, corr, mae, rsquared, num_points = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    rmse_train, rmse_val, rmse_train_std, rmse_val_std = [], [], [], []
    rmse_std, corr_std, mae_std, rsquared_std = [], [], [], []

    for key in rmse_stake.keys():
        gl_names.append(key.split("-")[0])
        stakes_full.append(GL_SHORT[key.split("-")[0]] + "-" +
                           key.split("-")[1])
        stakes.append(key.split("-")[1])
        num_points.append(len(pred_XG_stake[key]) / kfold)

        # Mean score
        rmse.append(np.mean(rmse_stake[key]) / (1000))
        corr.append(np.mean(pearson_stake[key]))
        mae.append(np.mean(mae_stake[key]) / (1000))
        rsquared.append(np.mean(rsquared_stake[key]))

        # Variance of score
        rmse_std.append(np.std(rmse_stake[key]) / (1000))
        if (type_pred == 'pred_XG') or (type_pred == 'RF') or (type_pred
                                                               == 'pred_LM'):
            rmse_train.append(np.mean(rmse_stake_train[key]) / (1000))
            rmse_val.append(np.mean(rmse_stake_val[key]) / (1000))
            rmse_train_std.append(np.std(rmse_stake_train[key]) / (1000))
            rmse_val_std.append(np.std(rmse_stake_val[key]) / (1000))
        corr_std.append(np.std(pearson_stake[key]))
        mae_std.append(np.std(mae_stake[key]) / (1000))
        rsquared_std.append(np.std(rsquared_stake[key]))

    data = {
        "glaciers": gl_names,
        "stakes": stakes,
        "stakes_full": stakes_full,
        "num_points": num_points,
        "rmse" + suffix_name: rmse,
        "corr" + suffix_name: corr,
        "mae" + suffix_name: mae,
        "r2" + suffix_name: rsquared,
        "rmse" + suffix_name + "_std": rmse_std,
        "corr" + suffix_name + "_std": corr_std,
        "mae" + suffix_name + "_std": mae_std,
        "r2" + suffix_name + "_std": rsquared_std,
    }
    if (type_pred == 'pred_XG') or (type_pred == 'RF') or (type_pred
                                                           == 'pred_LM'):
        data["rmse_train" + suffix_name + "_std"] = rmse_train_std
        data["rmse_val" + suffix_name + "_std"] = rmse_val_std
        data["rmse_train" + suffix_name] = rmse_train
        data["rmse_val" + suffix_name] = rmse_val
    df_metrics = pd.DataFrame(data)
    return df_metrics


def DiffMetrics(df_total_annual):
    # Comparison 1: compare xgb match ann with pdd ann+winter:
    df_total_annual['diff_rmse_a_aw'] = df_total_annual[
        'rmse_xgb'] - df_total_annual['rmse_pdd_a_aw']

    df_total_annual['diff_corr_a_aw'] = abs(df_total_annual['corr_xgb'] -
                                            df_total_annual['corr_pdd_a_aw'])
    df_total_annual['diff_r2_a_aw'] = df_total_annual[
        'r2_xgb'] - df_total_annual['r2_pdd_a_aw']

    df_total_annual['diff_mae_a_aw'] = df_total_annual[
        'mae_xgb'] - df_total_annual['mae_pdd_a_aw']

    # Comparison 2: compare xgb match ann with pdd match ann:
    df_total_annual['diff_rmse_a_a'] = df_total_annual[
        'rmse_xgb'] - df_total_annual['rmse_pdd_a']

    df_total_annual['diff_corr_a_a'] = abs(df_total_annual['corr_xgb'] -
                                           df_total_annual['corr_pdd_a'])
    df_total_annual['diff_r2_a_a'] = df_total_annual[
        'r2_xgb'] - df_total_annual['r2_pdd_a']

    df_total_annual['diff_mae_a_a'] = df_total_annual[
        'mae_xgb'] - df_total_annual['mae_pdd_a']

    # Comparison 3: compare xgb with winter matching ann MB with pdd ann+winter
    df_total_annual['diff_rmse_aw_aw'] = df_total_annual[
        'rmse_xgb_aw'] - df_total_annual['rmse_pdd_a_aw']

    df_total_annual['diff_corr_aw_aw'] = abs(df_total_annual['corr_xgb_aw'] -
                                             df_total_annual['corr_pdd_a_aw'])
    df_total_annual['diff_r2_aw_aw'] = df_total_annual[
        'r2_xgb_aw'] - df_total_annual['r2_pdd_a_aw']

    df_total_annual['diff_mae_aw_aw'] = df_total_annual[
        'mae_xgb_aw'] - df_total_annual['mae_pdd_a_aw']

    # Comparison 4: compare xgb with matching winter MB with pdd ann+winter
    df_total_annual['diff_rmse_w_w'] = df_total_annual[
        'rmse_xgb_w'] - df_total_annual['rmse_pdd_w']

    df_total_annual['diff_corr_w_w'] = abs(df_total_annual['corr_xgb_w'] -
                                           df_total_annual['corr_pdd_w'])
    df_total_annual['diff_r2_w_w'] = df_total_annual[
        'r2_xgb_w'] - df_total_annual['r2_pdd_w']

    df_total_annual['diff_mae_w_w'] = df_total_annual[
        'mae_xgb_w'] - df_total_annual['mae_pdd_w']

    return df_total_annual


def getDfMetrics(var_xg_monthly,
                 metrics_monthly,
                 NUM_FOLDS,
                 type_pred="pred_XG"):
    df_metrics_monthly = dfMetrics(var_xg_monthly[type_pred],
                                   metrics_monthly,
                                   kfold=NUM_FOLDS,
                                   type_pred=type_pred)
    
    df_metrics_monthly_corrected = dfMetricsCorrected(var_xg_monthly,
                                                  NUM_FOLDS,
                                                  type_pred=type_pred)
    if type_pred == 'pred_XG':
        df_metrics_monthly = df_metrics_monthly_corrected.merge(
            df_metrics_monthly[[
                'stakes_full', 'rmse_xgb', 'corr_xgb', 'mae_xgb', 'r2_xgb',
                'rmse_xgb_std', 'corr_xgb_std', 'mae_xgb_std', 'r2_xgb_std'
            ]],
            on='stakes_full',
            suffixes=('_full', '_folds')).sort_values(by='stakes_full')
    elif type_pred == 'annual_pred_PDD':
        df_metrics_monthly = df_metrics_monthly_corrected.merge(
            df_metrics_monthly[[
                'stakes_full', 'rmse_pdd_a', 'corr_pdd_a', 'mae_pdd_a', 'r2_pdd_a',
                'rmse_pdd_a_std', 'corr_pdd_a_std', 'mae_pdd_a_std', 'r2_pdd_a_std'
            ]],
            on='stakes_full',
            suffixes=('_full', '_folds')).sort_values(by='stakes_full')
    
    return df_metrics_monthly


def get_temp_Df_metrics(var_monthly,
                        metrics_monthly,
                        var_seasonal,
                        metrics_seasonal,
                        var_half_year,
                        metrics_half_year,
                        var_annual,
                        metrics_annual,
                        num_folds,
                        type_pred='Lasso',
                        pred='pred_LM',
                        corrected=True):
    # Create metric dfs for Lasso:
    if corrected:
        df_metrics_monthly = dfMetricsCorrected(
            var_monthly, kfold=num_folds,
            type_pred=type_pred).sort_values(by='stakes_full')

        df_metrics_seasonal = dfMetricsCorrected(
            var_seasonal, kfold=num_folds,
            type_pred=type_pred).sort_values(by='stakes_full')

        df_metrics_half = dfMetricsCorrected(
            var_half_year, kfold=num_folds,
            type_pred=type_pred).sort_values(by='stakes_full')

        df_metrics_annual = dfMetricsCorrected(
            var_annual, kfold=num_folds,
            type_pred=type_pred).sort_values(by='stakes_full')
    else:
        df_metrics_monthly = dfMetrics(
            var_monthly[pred],
            metrics_monthly,
            kfold=num_folds,
            type_pred=type_pred).sort_values(by='stakes_full')

        df_metrics_seasonal = dfMetrics(
            var_seasonal[pred],
            metrics_seasonal,
            kfold=num_folds,
            type_pred=type_pred).sort_values(by='stakes_full')

        df_metrics_half = dfMetrics(
            var_half_year[pred],
            metrics_half_year,
            kfold=num_folds,
            type_pred=type_pred).sort_values(by='stakes_full')

        df_metrics_annual = dfMetrics(
            var_annual[pred],
            metrics_annual,
            kfold=num_folds,
            type_pred=type_pred).sort_values(by='stakes_full')

    return df_metrics_annual, df_metrics_seasonal, df_metrics_half, df_metrics_monthly


# Pdd matching only annual MB (to be fair with ML models):
# Create metrics grouped by glaciers and stakes:


def Df_metrics_plot(df_metrics_pdd, df_metrics_lasso_monthly,
                    df_metrics_monthly, df_metrics_annual, df_metrics_seasonal,
                    df_metrics_half, df_metrics_best):
    df_pdd_grouped_a = df_metrics_pdd[[
        'glaciers', 'stakes', 'stakes_full', 'rmse_pdd_a_folds',
        'mae_pdd_a_folds', 'corr_pdd_a_folds', 'r2_pdd_a_folds',
        'rmse_pdd_a_full', 'mae_pdd_a_full', 'corr_pdd_a_full', 'r2_pdd_a_full'
    ]].groupby(['glaciers', 'stakes', 'stakes_full'
                ]).mean().reset_index().sort_values(by='stakes_full')

    df_metrics_grouped_lasso = df_metrics_lasso_monthly.groupby(
        ['glaciers', 'stakes',
         'stakes_full']).mean().reset_index().sort_values(by='stakes_full')

    df_metrics_grouped_xgb = df_metrics_monthly.groupby(
        ['glaciers', 'stakes',
         'stakes_full']).mean().reset_index().sort_values(by='stakes_full')

    df_metrics_grouped_xgb_annual = df_metrics_annual.groupby(
        ['glaciers', 'stakes',
         'stakes_full']).mean().reset_index().sort_values(by='stakes_full')

    df_metrics_grouped_xgb_seasonal = df_metrics_seasonal.groupby(
        ['glaciers', 'stakes',
         'stakes_full']).mean().reset_index().sort_values(by='stakes_full')

    df_metrics_grouped_xgb_half = df_metrics_half.groupby(
        ['glaciers', 'stakes',
         'stakes_full']).mean().reset_index().sort_values(by='stakes_full')

    df_metrics_grouped_xgb_bestvar = df_metrics_best.groupby(
        ['glaciers', 'stakes',
         'stakes_full']).mean().reset_index().sort_values(by='stakes_full')

    # Merge dataframes:
    df_metrics_plot = df_pdd_grouped_a.merge(df_metrics_grouped_lasso[[
        'corr_lasso', 'rmse_lasso', 'r2_lasso', 'glaciers', 'stakes'
    ]],
                                             on=['glaciers',
                                                 'stakes']).reset_index()

    df_metrics_plot = df_metrics_plot.merge(df_metrics_grouped_xgb[[
        'corr_xgb_folds', 'rmse_xgb_folds', 'r2_xgb_folds', 'mae_xgb_folds',
        'corr_xgb_full', 'rmse_xgb_full', 'r2_xgb_full', 'corr_xgb_std',
        'rmse_xgb_std', 'r2_xgb_std', 'glaciers', 'stakes', 'num_points'
    ]],
                                            on=['glaciers',
                                                'stakes']).reset_index()

    df_metrics_plot = df_metrics_plot.drop(['level_0', 'index'], axis=1)

    # Annual XGBoost:
    df_metrics_plot = df_metrics_plot.merge(df_metrics_grouped_xgb_annual[[
        'corr_xgb_folds', 'rmse_xgb_folds', 'r2_xgb_folds', 'mae_xgb_folds',
        'corr_xgb_full', 'rmse_xgb_full', 'r2_xgb_full', 'corr_xgb_std',
        'rmse_xgb_std', 'r2_xgb_std', 'glaciers', 'stakes', 'num_points'
    ]],
                                            on=['glaciers', 'stakes'],
                                            suffixes=('',
                                                      '_ann')).reset_index()

    df_metrics_plot = df_metrics_plot.merge(df_metrics_grouped_xgb_seasonal[[
        'corr_xgb_folds', 'rmse_xgb_folds', 'r2_xgb_folds', 'mae_xgb_folds',
        'corr_xgb_full', 'rmse_xgb_full', 'r2_xgb_full', 'corr_xgb_std',
        'rmse_xgb_std', 'r2_xgb_std', 'glaciers', 'stakes', 'num_points'
    ]],
                                            on=['glaciers', 'stakes'],
                                            suffixes=('',
                                                      '_seas')).reset_index()
    df_metrics_plot = df_metrics_plot.drop(['level_0', 'index'], axis=1)

    df_metrics_plot = df_metrics_plot.merge(df_metrics_grouped_xgb_half[[
        'corr_xgb_folds', 'rmse_xgb_folds', 'r2_xgb_folds', 'mae_xgb_folds',
        'corr_xgb_full', 'rmse_xgb_full', 'r2_xgb_full', 'corr_xgb_std',
        'rmse_xgb_std', 'r2_xgb_std', 'glaciers', 'stakes', 'num_points'
    ]],
                                            on=['glaciers', 'stakes'],
                                            suffixes=('',
                                                      '_halfy')).reset_index()
    df_metrics_plot = df_metrics_plot.drop(['index'], axis=1)

    # XGBoost++:
    df_metrics_plot = df_metrics_plot.merge(df_metrics_grouped_xgb_bestvar[[
        'corr_xgb_folds', 'rmse_xgb_folds', 'r2_xgb_folds', 'mae_xgb_folds',
        'corr_xgb_full', 'rmse_xgb_full', 'r2_xgb_full', 'corr_xgb_std',
        'rmse_xgb_std', 'r2_xgb_std', 'glaciers', 'stakes', 'num_points'
    ]],
                                            suffixes=('', '_bestvar'),
                                            on=['glaciers',
                                                'stakes']).reset_index()
    df_metrics_plot = df_metrics_plot.drop(['index'], axis=1)

    # Change to m w.e.
    cols = [
        'rmse_lasso', 'rmse_xgb_folds', 'rmse_xgb_folds_ann',
        'rmse_xgb_folds_halfy', 'rmse_xgb_folds_seas',
        'rmse_xgb_folds_bestvar', 'rmse_pdd_a_folds', 'rmse_xgb_full',
        'rmse_xgb_full_ann', 'rmse_xgb_full_halfy', 'rmse_xgb_full_seas',
        'rmse_xgb_full_bestvar', 'rmse_pdd_a_full', 'rmse_xgb_std_bestvar',
        'corr_xgb_std_bestvar', 'rmse_xgb_std_bestvar', 'r2_xgb_std_bestvar'
    ]
    for col in cols:
        df_metrics_plot[col] = df_metrics_plot[col] / 1000

    # Differences between models
    # RMSE:
    df_metrics_plot['diff_rmse_lasso'] = (df_metrics_plot['rmse_lasso'] -
                                          df_metrics_plot['rmse_pdd_a_folds'])
    df_metrics_plot['diff_r2_lasso_folds'] = df_metrics_plot[
        'r2_lasso'] - df_metrics_plot['r2_pdd_a_folds']
    suffix = 'folds'
    cols_diff = [
        'diff_rmse_xgb',
        'diff_rmse_xgb_ann',
        'diff_rmse_xgb_halfy',
        'diff_rmse_xgb_seas',
        'diff_rmse_xgb_bestvar',
    ]
    cols = [
        'rmse_xgb_folds', 'rmse_xgb_folds_ann', 'rmse_xgb_folds_halfy',
        'rmse_xgb_folds_seas', 'rmse_xgb_folds_bestvar'
    ]
    for i, col in enumerate(cols_diff):
        df_metrics_plot[col + '_' +
                        suffix] = (df_metrics_plot[cols[i]] -
                                   df_metrics_plot['rmse_pdd_a_' + suffix])
    # R2:
    cols_diff = [
        'diff_r2_xgb',
        'diff_r2_xgb_ann',
        'diff_r2_xgb_halfy',
        'diff_r2_xgb_seas',
        'diff_r2_xgb_bestvar',
    ]
    cols = [
        'r2_xgb_folds', 'r2_xgb_folds_ann', 'r2_xgb_folds_halfy',
        'r2_xgb_folds_seas', 'r2_xgb_folds_bestvar'
    ]
    for i, col in enumerate(cols_diff):
        df_metrics_plot[col + '_' +
                        suffix] = (df_metrics_plot[cols[i]] -
                                   df_metrics_plot['r2_pdd_a_' + suffix])

    suffix = 'full'
    cols_diff = [
        'diff_rmse_xgb',
        'diff_rmse_xgb_ann',
        'diff_rmse_xgb_halfy',
        'diff_rmse_xgb_seas',
        'diff_rmse_xgb_bestvar',
    ]
    cols = [
        'rmse_xgb_folds', 'rmse_xgb_folds_ann', 'rmse_xgb_folds_halfy',
        'rmse_xgb_folds_seas', 'rmse_xgb_folds_bestvar'
    ]
    for i, col in enumerate(cols_diff):
        df_metrics_plot[col + '_' +
                        suffix] = (df_metrics_plot[cols[i]] -
                                   df_metrics_plot['rmse_pdd_a_' + suffix])
    # R2:
    cols_diff = [
        'diff_r2_xgb',
        'diff_r2_xgb_ann',
        'diff_r2_xgb_halfy',
        'diff_r2_xgb_seas',
        'diff_r2_xgb_bestvar',
    ]
    cols = [
        'r2_xgb_folds', 'r2_xgb_folds_ann', 'r2_xgb_folds_halfy',
        'r2_xgb_folds_seas', 'r2_xgb_folds_bestvar'
    ]
    for i, col in enumerate(cols_diff):
        df_metrics_plot[col + '_' +
                        suffix] = (df_metrics_plot[cols[i]] -
                                   df_metrics_plot['r2_pdd_a_' + suffix])

    return df_metrics_plot


def add_rmse_r2(df_metrics, var_xg):
    rmse_real, rsquared_real = [], []
    for i in range(len(df_metrics)):
        stakeName = df_metrics.iloc[i].glaciers + '_' + df_metrics.iloc[
            i].stakes
        pred = var_xg['pred_XG'][stakeName]
        target = var_xg['feat_test'][stakeName]['target']
        rmse_real.append(mean_squared_error(pred, target, squared=False))
        rsquared_real.append(r_squared(pred, target))

    df_metrics['rmse_xgb'] = rmse_real
    df_metrics['r2_xgb'] = rsquared_real
    return df_metrics


def add_rmse_r2_pdd(df_metrics, var_pdd_a):
    rmse_real, rsquared_real = [], []
    for i in range(len(df_metrics)):
        stakeName = df_metrics.iloc[i].glaciers + '_' + df_metrics.iloc[
            i].stakes
        target = var_pdd_a['feat_test'][stakeName]['target_a']
        pred = var_pdd_a['annual_pred_PDD'][stakeName]
        rmse_real.append(mean_squared_error(target, pred, squared=False))
        rsquared_real.append(r_squared(target, pred))

    df_metrics['rmse_xgb'] = rmse_real
    df_metrics['r2_xgb'] = rsquared_real
    return df_metrics


def calcDiff(df_metrics, df_metrics_pdd):
    df_diff = df_metrics.merge(df_metrics_pdd[[
        'stakes_full', f'rmse_pdd_a_full', f'mae_pdd_a_full',
        f'corr_pdd_a_full', f'r2_pdd_a_full'
    ]],
                               on='stakes_full')
    df_diff[f'diff_rmse_xgb_full'] = (df_diff[f'rmse_xgb_full'] -
                                          df_diff[f'rmse_pdd_a_full'])

    df_diff[f'diff_corr_xgb_full'] = (df_diff[f'corr_xgb_full'] -
                                          df_diff[f'corr_pdd_a_full'])

    df_diff[f'diff_mae_xgb_full'] = (df_diff[f'mae_xgb_full'] -
                                         df_diff[f'mae_pdd_a_full'])

    df_diff[f'diff_mae_wrt_to_std'] = df_diff[
        f'diff_mae_xgb_full'] / df_diff['std_obs']

    return df_diff
