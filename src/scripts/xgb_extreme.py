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
import seaborn as sns
from scripts.xgb_helpers import *
from scripts.xgb_model import *
from scripts.xgb_model_varcomb import *
from scripts.xgb_metrics import *
from scripts.xgb_input import *
from scripts.stakes_processing import *
from scripts.PDD_model_calibration import *


def createSingleInput(best_combi,
                      weights_t2m,
                      weights_tp,
                      stakeName,
                      input_type,
                      vars_,
                      log=False,
                      unseen=False):
    # One stake
    # Read MB data:
    df_stake = read_stake_csv(path_glacattr, f'{stakeName}_mb.csv')

    # Read climate data (pre-processed before to be at location of stake)
    if input_type == "ERA5-Land":
        # Read corresponding era 5 land values for this stake:
        xr_temppr = xr.open_dataset(path_era5_stakes +
                                    f"{stakeName}_mb_full.nc").sortby("time")
    if input_type == "MeteoSuisse" and unseen == False:
        # Read corresponding meteo suisse values for this stake:
        xr_temppr = xr.open_dataset(path_MS +
                                    f"{stakeName}_mb_full.nc").sortby("time")
    if input_type == "MeteoSuisse" and unseen == True:
        # Read corresponding meteo suisse values for this stake:
        xr_temppr = xr.open_dataset(path_MS_full +
                                    f"{stakeName}_mb_full.nc").sortby("time")
    begin_xr = pd.to_datetime(xr_temppr["time"].values[0]).year
    end_xr = pd.to_datetime(xr_temppr["time"].values[-1]).year

    # Cut MB data to same years as xr climate:
    # df_stake_cut = cutStake(df_stake, begin_xr, end_xr)
    df_stake_cut = df_stake

    # Remove cat 0 (only modelled MB)
    target_DF = df_stake_cut[df_stake_cut.vaw_id > 0]

    inputDF = createInputDF_varcomb(best_combi[0][0],
                                    best_combi[0][1],
                                    df_stake_cut,
                                    xr_temppr,
                                    weights_t2m,
                                    weights_tp,
                                    month_val=MONTH_VAL,
                                    input_type=input_type,
                                    match="annual",
                                    vars_=vars_,
                                    unseen=unseen)
    # Create target:
    target_years = [d.year for d in target_DF.date_fix1]
    target = target_DF[["date_fix1", "b_a_fix", "b_w_fix"]]
    target["years"] = target_years
    target.set_index("years", inplace=True)
    target = target.loc[target_years]

    # Cut inputDF years to same as in target:
    intersec = intersection(inputDF.index, target_years)
    if unseen == False:
        inputDF = inputDF.loc[intersec]
        target = target.loc[intersec]
    else:
        # start target at MS time - hydr. year of 1962
        target = target.loc[begin_xr + 1:end_xr]
    if log:
        print('Input DF shape: {}'.format(inputDF.shape))
        print(f'Target shape: {target.shape}')

    return inputDF, target, xr_temppr


def createSingleInputArraysExtreme(test_years,
                                   target,
                                   inputDF,
                                   log=False,
                                   seed=SEED):
    seed_all(seed)
    # Create test and training sets:
    # Remove extreme years for testing:
    test_index = [target.index.get_loc(key)
                  for key in test_years]  # 2022 and 2023
    rest_index = np.setdiff1d(np.arange(len(target)), test_index)

    indices = {"test": test_index, "train2": rest_index}

    # Create test and training features

    sss = ShuffleSplit(n_splits=1, test_size=0.2)
    # Create training features
    X_train2, y_train2, time_train2 = getXYTime(inputDF,
                                                target,
                                                indices["train2"],
                                                type_target="b_a_fix")

    # Split rest data into train and validation:
    sss.get_n_splits(X_train2, y_train2)
    train_index, val_index = next(sss.split(X_train2, y_train2))
    X_train, y_train, time_train = getXYTime(inputDF,
                                             target,
                                             train_index,
                                             type_target="b_a_fix")
    X_val, y_val, time_val = getXYTime(inputDF,
                                       target,
                                       val_index,
                                       type_target="b_a_fix")

    # Create test set:
    X_test, y_test, time_test = getXYTime(inputDF,
                                          target,
                                          indices["test"],
                                          type_target="b_a_fix")

    X = {"test": X_test, "train2": X_train2, "train": X_train, "val": X_val}
    y = {"test": y_test, "train2": y_train2, "train": y_train, "val": y_val}
    time = {
        "test": time_test,
        "train2": time_train2,
        "train": time_train,
        "val": time_val,
    }
    if log:
        print('-' * 50)
        print('Test years:', time["test"].values)
        print('Shapes of X test:', X["test"].shape)
        print('Shapes of X train:', X["train"].shape)
        print('Shapes of X validation:', X["val"].shape)

    return X, y, time


def applySingleXGBoost(
    X,
    y,
    param_grid,
    objective='reg:absoluteerror',  # training loss
    log=False,
    seed=SEED,
    grid_search=True,
    custom_params_RF={},
):
    seed_all(seed)
    # Apply XGBoost model
    # Grid search:
    eval_set = [(X["train"], y["train"]), (X["val"], y["val"])]
    eval_metrics = ["rmse", "mae"]

    # return best parameters and best estimator
    if log:
        print('-' * 50)
        print('Running grid search:')
    if grid_search:
        xgboost, best_params_, params = RGS(X['train2'],
                                            y["train2"],
                                            eval_set,
                                            param_grid,
                                            eval_metrics,
                                            objective=objective,
                                            log_=log)
        # Fit best model:
        params_RF = {**best_params_, **params}
    else:
        params_RF = custom_params_RF
        params_RF["gpu_id"] = 0
        params_RF["tree_method"] = "gpu_hist"
        params_RF["objective"] = objective
        params_RF["random_state"] = seed
        xgboost = xgb.XGBRegressor(**params_RF)
        xgboost.fit(
            X["train"],
            y["train"],
            eval_metric=eval_metrics,
            eval_set=eval_set,
            verbose=False,
            early_stopping_rounds=10,
        )
    # Evaluate:
    # Best HP rsults:
    num_trees = params_RF["n_estimators"]
    early_stopping_rounds = round(0.1 * num_trees)

    # gets training and validation loss of best estimator:
    results = xgboost.evals_result()
    best_iteration = xgboost.best_iteration
    if log:
        print(
            f"Nb of trees: {num_trees} and early stopping: {early_stopping_rounds}\nBest iteration: {best_iteration}"
        )

    # Losses:
    val_loss = results["validation_1"]
    train_loss = results["validation_0"]
    epochs = len(results["validation_1"][eval_metrics[0]])

    # Parameters:
    # lr = params_RF['learning_rate']
    # nest = params_RF['n_estimators']
    # max_depth = params_RF['max_depth']

    # Prediction:
    predictions_XG = xgboost.predict(X["test"],
                                     iteration_range=[0, best_iteration])

    return predictions_XG, val_loss, train_loss, epochs, best_iteration, params_RF


# --------------- PDD
def createSingleInputArraysExtreme_PDD(target, test_years, seed=SEED):
    # Apply PDD:
    seed_all(seed)

    # Create test and training sets:
    # Remove extreme years for testing:
    test_index = [target.index.get_loc(key)
                  for key in test_years]  # 2022 and 2023
    rest_index = np.setdiff1d(np.arange(len(target)), test_index)
    indices = {"test": test_index, "train2": rest_index}

    # Separate into training and test
    ann_mb_test = target["b_a_fix"].iloc[indices['test']].values  # [mm w.e.]
    w_mb_test = target["b_w_fix"].iloc[indices['test']].values  # [mm w.e.]
    test_time = [d.year for d in target.iloc[indices['test']].date_fix1
                 ]  # take year to be second part of hydrological year

    ann_mb_train = target["b_a_fix"].iloc[
        indices['train2']].values  # [mm w.e.]
    w_mb_train = target["b_w_fix"].iloc[indices['train2']].values  # [mm w.e.]
    train_time = [d.year for d in target.iloc[indices['train2']].date_fix1
                  ]  # take year to be second part of hydrological year

    time = {'train2': train_time, 'test': test_time}
    ann_mb = {'train2': ann_mb_train, 'test': ann_mb_test}
    w_mb = {'train2': w_mb_train, 'test': w_mb_test}
    return ann_mb, w_mb, time


def applySinglePDD(stake, xr_full, time, ann_mb, w_mb, dTdz_stakes, dPdz,
                   c_prec, DDFsnow, DDFice, DDFsnow_range, c_prec_range,
                   inital_params, constants, input_type):
    # Search for best PDD parameters
    # Run PDD model on train years and report parameters for all years:
    c_year, DDFsnow_year, DDFice_year, w_matched_year, an_matched_year = PDDModel_paramsearch(
        f'{stake}_mb.csv',
        xr_full,
        time['train2'],
        ann_mb['train2'],
        w_mb['train2'],
        dTdz_stakes,
        dPdz,
        c_prec,
        DDFsnow,
        DDFice,
        DDFsnow_range,
        c_prec_range,
        inital_params,
        constants,
        match_winter=True,
        input_type=input_type,
        log=False,
    )

    # Predictions:
    # Take average parameters and make predictions on test years:
    c_prec_avg, DDFsnow_avg, DDFice_avg = (
        np.mean(c_year),
        np.mean(DDFsnow_year),
        np.mean(DDFsnow_year) * 2,
    )
    pred_mb_w, pred_mb_ann = PDD_prediction(  # [mm w.e.]
        f'{stake}_mb.csv',
        xr_full,
        dTdz_stakes,
        time['test'],
        c_prec_avg,
        DDFsnow_avg,
        DDFice_avg,
        constants,
        inital_params,
        input_type=input_type)

    return pred_mb_w, pred_mb_ann, c_year, DDFsnow_year


# ------------------------ Plotting
def addError(dfStake_xgb, color_xgb, marker_xgb, ax):
    dfStake_xgb['error'] = dfStake_xgb['pred'] - dfStake_xgb['obs']
    lowlims = dfStake_xgb["error"] > 0
    uplims = dfStake_xgb["error"] <= 0

    plotline1, caplines1, barlinecols1 = ax.errorbar(
        x=dfStake_xgb["time"],
        y=dfStake_xgb["obs"],
        yerr=dfStake_xgb["error"].abs(),
        uplims=uplims,
        lolims=lowlims,
        linewidth=0.5,
        linestyle="-",
        alpha=1,
        color="grey",
        elinewidth=1,
        ecolor=color_xgb,
    )

    caplines1[0].set_marker(marker_xgb)
    caplines1[0].set_markersize(6)
    barlinecols1[0].set_linestyle("--")
    if len(caplines1) > 1:
        caplines1[1].set_marker(marker_xgb)
        caplines1[1].set_markersize(6)


def AssemblePred_xtrm(stake_old,
                      stake_new,
                      vars_,
                      weights_t2m,
                      weights_tp,
                      dfPred_pdd,
                      dfPrediction_xgb,
                      best_combi,
                      input_type,
                      test_index=[2022, 2023]):
    # stakeName = stake_old.split('_')[0] + '_' + stake_old.split('_')[1]
    # XGB
    inputDF, target, xr_full = createSingleInput(best_combi,
                                                 weights_t2m,
                                                 weights_tp,
                                                 stake_old,
                                                 input_type=input_type,
                                                 vars_=vars_,
                                                 log=False)

    # Separate into training and testing:
    X_xgb, y_xgb, time_xgb = createSingleInputArraysExtreme(test_index,
                                                            target,
                                                            inputDF,
                                                            log=False)

    predictions_XG = dfPrediction_xgb[dfPrediction_xgb.stake ==
                                      stake_new][test_index].values[0]
    dfTrain_xgb = pd.DataFrame({
        'time': time_xgb["train2"],
        'obs': y_xgb["train2"] / (1000),
        'type': np.tile('obs', len(y_xgb["train2"]))
    })

    dfPrediction_xgb = pd.DataFrame({
        'time':
        time_xgb["test"],
        'obs':
        y_xgb["test"] / (1000),
        'pred':
        predictions_XG / (1000),
        'type':
        np.tile('pred_xgb', len(y_xgb["test"]))
    })

    dfStake_xgb = pd.concat([dfTrain_xgb,
                             dfPrediction_xgb]).sort_values('time')

    # PDD
    ann_mb, w_mb, time_pdd = createSingleInputArraysExtreme_PDD(target,
                                                                test_index,
                                                                seed=SEED)
    dfTrain_PDD = pd.DataFrame({
        'time': time_pdd["train2"],
        'obs': ann_mb["train2"] / (1000),
        'type': np.tile('obs', len(ann_mb["train2"]))
    })

    pred_pdd_ann = dfPred_pdd[dfPred_pdd.stake ==
                              stake_new][test_index].values[0]
    dfPrediction_PDD = pd.DataFrame({
        'time':
        time_pdd['test'],
        'obs':
        ann_mb["test"] / (1000),
        'pred':
        pred_pdd_ann / (1000),
        'type':
        np.tile('pred_pdd', len(ann_mb["test"]))
    })
    dfStake_pdd = pd.concat([dfTrain_PDD,
                             dfPrediction_PDD]).sort_values('time')

    return dfStake_xgb, dfTrain_xgb, dfPrediction_xgb, dfStake_pdd, dfPrediction_PDD


def plotPredStake(stake_new,
                  stake_old,
                  weights_t2m,
                  weights_tp,
                  dfMetrics_pdd,
                  dfMetrics_xgb,
                  dfPred_pdd,
                  dfPred_xgb,
                  best_combi,
                  ax,
                  input_type,
                  color_xgb,
                  color_tim,
                  marker_xgb,
                  marker_tim,
                  vars_,
                  test_years=[2022, 2023]):
    dfStake_xgb, dfTrain_xgb, dfPrediction_xgb, dfStake_pdd, dfPrediction_PDD = AssemblePred_xtrm(
        stake_new=stake_new,
        stake_old=stake_old,
        vars_=vars_,
        weights_t2m=weights_t2m,
        weights_tp=weights_tp,
        dfPred_pdd=dfPred_pdd,
        dfPrediction_xgb=dfPred_xgb,
        best_combi=best_combi,
        input_type=input_type,
        test_index=test_years)

    mae_pdd_2022, mae_pdd_2023 = dfMetrics_pdd[
        dfMetrics_pdd.stake == stake_new][['MAE_2022', 'MAE_2023']].values[0]
    mae_2022, mae_2023, std_obs = dfMetrics_xgb[
        dfMetrics_xgb.stake == stake_new][['MAE_2022', 'MAE_2023',
                                           'std_obs']].values[0]

    sns.scatterplot(x='time',
                    y='obs',
                    data=dfTrain_xgb,
                    label='',
                    color='grey',
                    alpha=0.8,
                    marker='.',
                    ax=ax)
    sns.lineplot(x='time',
                 y='obs',
                 data=dfStake_xgb,
                 label='Observed',
                 color='grey',
                 alpha=0.8,
                 ax=ax)

    addError(dfStake_xgb, color_xgb, marker_xgb, ax)
    addError(dfStake_pdd, color_tim, marker_tim, ax)

    legend_text = "\n".join((
        r"$\mathrm{MAE_{miniML}(2022)}=%.2f, \mathrm{MAE_{PDD}(2022)}=%.2f, \mathrm{std_{obs}}=%.2f$"
        % (mae_2022, mae_pdd_2022, std_obs),
        r"$\mathrm{MAE_{miniML}(2023)}=%.2f, \mathrm{MAE_{PDD}(2023)}=%.2f$" %
        (
            mae_2023,
            mae_pdd_2023,
        ),
        # (r"$\mathrm{std_{obs}}=%.2f$" % (
        #     std_obs,
        # )),
    ))
    ax.text(0.02,
            0.02,
            legend_text,
            transform=ax.transAxes,
            verticalalignment="bottom",
            fontsize=14)

    # set ylim to be 10% of min observed value
    ax.set_ylim(bottom=dfStake_xgb['obs'].min() * 1.5)


def plotDiffMetrics(dfDiffMetrics, color_diff_xgbplus, marker_tim, color_tim,
                    marker_xgb, color_xgbplus):
    fig = plt.figure(figsize=(15, 10))
    M, N = 2, 2

    alpha = 0.8
    fontsize_title = 22

    # Diff MAE:
    ax1 = plt.subplot(M, N, 1)
    g = sns.barplot(
        dfDiffMetrics,
        x='stakes_full',
        y=f'diff_mae',
        ax=ax1,
        dodge=False,
        color=color_diff_xgbplus,
        alpha=alpha,
    )
    h, l = g.get_legend_handles_labels()
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
    ax1.legend(
        h,
        l,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        ncol=2,
        borderaxespad=0.0,
    )
    ax1.set_ylabel('[m w.e. a$^{-1}$]', fontsize=18)
    ax1.set_title('$\mathrm{MAE(XGB_{opt.seas.})-MAE(TIM)}$',
                  fontsize=fontsize_title)

    ax2 = plt.subplot(M, N, 2, sharey=ax1)
    g = sns.barplot(
        dfDiffMetrics,
        x='stakes_full',
        y=f'diff_rmse',
        ax=ax2,
        dodge=False,
        alpha=alpha,
        color=color_diff_xgbplus,
    )
    h, l = g.get_legend_handles_labels()
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)
    ax2.set_title(
        '$\mathrm{RMSE}(\mathrm{XGB_{opt.seas.}})-\mathrm{RMSE}(\mathrm{TIM})$',
        fontsize=fontsize_title)
    ax2.set_ylabel('[m w.e. a$^{-1}$]', fontsize=18)

    ax3 = plt.subplot(M, N, 3)
    sns.scatterplot(dfDiffMetrics,
                    x='stakes_full',
                    y=f'MAE_pdd',
                    ax=ax3,
                    alpha=alpha,
                    marker=marker_tim,
                    color=color_tim)
    g = sns.scatterplot(dfDiffMetrics,
                        x='stakes_full',
                        y=f'MAE_xgb',
                        ax=ax3,
                        alpha=alpha,
                        color=color_xgbplus,
                        marker=marker_xgb,
                        s=200)
    h, l = g.get_legend_handles_labels()
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    ax3.set_title(
        '$\mathrm{RMSE}(\mathrm{XGB_{opt.seas.}})$ vs $\mathrm{RMSE}(\mathrm{TIM})$',
        fontsize=fontsize_title)
    ax3.set_ylabel('[m w.e. a$^{-1}$]', fontsize=18)
    ax3.grid(axis='both', linestyle='--')

    ax4 = plt.subplot(M, N, 4, sharey=ax3)
    sns.scatterplot(dfDiffMetrics,
                    x='stakes_full',
                    y=f'RMSE_pdd',
                    ax=ax4,
                    alpha=alpha,
                    marker=marker_tim,
                    color=color_tim)
    g = sns.scatterplot(dfDiffMetrics,
                        x='stakes_full',
                        y=f'RMSE_xgb',
                        ax=ax4,
                        alpha=alpha,
                        color=color_xgbplus,
                        marker=marker_xgb,
                        s=200)
    h, l = g.get_legend_handles_labels()
    ax4.grid(axis='both', linestyle='--')
    ax4.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    ax4.legend(
        h,
        l,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        ncol=2,
        borderaxespad=0.0,
    )
    ax4.set_ylabel('[m w.e. a$^{-1}$]', fontsize=18)
    ax4.set_title('$\mathrm{MAE(XGB_{opt.seas.})}$ vs $\mathrm{MAE(TIM)}$',
                  fontsize=fontsize_title)
    ax4.set_ylim(bottom=0)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend([], [], frameon=False)
        ax.set_xlabel('')
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)

    plt.tight_layout()


def CreateDiffMetrics_xtr(dfMetrics_xgb, dfMetrics_pdd):
    # Plot difference per stakes_full:
    dfDiffMetrics = pd.merge(dfMetrics_xgb.drop(['Unnamed: 0'], axis=1),
                             dfMetrics_pdd.drop(['Unnamed: 0'], axis=1),
                             on='stake',
                             suffixes=('_xgb', '_pdd'))
    dfDiffMetrics[
        'diff_mae'] = dfDiffMetrics['MAE_xgb'] - dfDiffMetrics['MAE_pdd']
    dfDiffMetrics[
        'diff_rmse'] = dfDiffMetrics['RMSE_xgb'] - dfDiffMetrics['RMSE_pdd']

    dfDiffMetrics['diff_mae_2022'] = dfDiffMetrics[
        'MAE_2022_xgb'] - dfDiffMetrics['MAE_2022_pdd']
    dfDiffMetrics['diff_rmse_2022'] = dfDiffMetrics[
        'RMSE_2022_xgb'] - dfDiffMetrics['RMSE_2022_pdd']

    dfDiffMetrics['diff_mae_2023'] = dfDiffMetrics[
        'MAE_2023_xgb'] - dfDiffMetrics['MAE_2023_pdd']
    dfDiffMetrics['diff_rmse_2023'] = dfDiffMetrics[
        'RMSE_2023_xgb'] - dfDiffMetrics['RMSE_2023_pdd']

    dfDiffMetrics['stakes_full'] = dfDiffMetrics.stake.apply(
        lambda x: GL_SHORT[x.split('-')[0]]) + '-' + dfDiffMetrics.stake.apply(
            lambda x: x.split('-')[1])

    dfDiffMetrics[f'diff_mae_2023_wrt_to_std'] = dfDiffMetrics[
        f'diff_mae_2023'] / np.abs(dfDiffMetrics['std_obs'])
    dfDiffMetrics[f'diff_mae_2022_wrt_to_std'] = dfDiffMetrics[
        f'diff_mae_2022'] / np.abs(dfDiffMetrics['std_obs'])

    return dfDiffMetrics


def GapFilling(stake, stake_old, input_type, param_grid, custom_params_RF, grid_search = True):
    vars_ = ['t2m', 'tp']
    best_combi = [(['t2m_May', 't2m_June', 't2m_July', 't2m_Aug'], [
        'tp_Oct',
        'tp_Nov',
        'tp_Dec',
        'tp_Jan',
        'tp_Feb',
    ])]

    weights_t2m = np.ones(len(best_combi[0][0]))
    weights_tp = np.ones(len(best_combi[0][1]))

    # get test index:
    inputDF, target, xr_temppr = createSingleInput(best_combi,
                                                   weights_t2m,
                                                   weights_tp,
                                                   stake_old,
                                                   input_type,
                                                   vars_,
                                                   log=False,
                                                   unseen=True)
    test_index = Diff(list(inputDF.index.values), list(target.index.values))
    test_index.sort()
    print(stake, ':')
    if len(test_index) > 0:
        print('Predicting for missing years:', test_index)
        # print('Input DF shape: {}'.format(inputDF.shape))
        # print(f'Target shape: {target.shape}')

        # Separate inputDF into training and testing:
        inputDF_test = inputDF.loc[test_index]
        inputDF_train = inputDF.drop(test_index)

        # Create arrays:
        # Create test and training sets:
        rest_index = np.arange(len(target))
        indices = {"test": test_index, "train2": rest_index}

        # Create training features
        X_train2, y_train2, time_train2 = getXYTime(inputDF_train,
                                                    target,
                                                    indices["train2"],
                                                    type_target="b_a_fix")
        # Split rest data into train and validation:
        sss = ShuffleSplit(n_splits=1, test_size=0.2)
        sss.get_n_splits(X_train2, y_train2)
        train_index, val_index = next(sss.split(X_train2, y_train2))
        X_train, y_train, time_train = getXYTime(inputDF_train,
                                                 target,
                                                 train_index,
                                                 type_target="b_a_fix")
        X_val, y_val, time_val = getXYTime(inputDF_train,
                                           target,
                                           val_index,
                                           type_target="b_a_fix")

        # Test set
        X_test, time_test = inputDF_test.values, inputDF_test.index

        X = {
            "test": X_test,
            "train2": X_train2,
            "train": X_train,
            "val": X_val
        }
        y = {"train2": y_train2, "train": y_train, "val": y_val}
        time = {
            "test": time_test,
            "train2": time_train2,
            "train": time_train,
            "val": time_val,
        }

        # apply XGBoost model:
        predictions_XG, val_loss, train_loss, epochs, best_iteration, params_RF = applySingleXGBoost(
            X, y, param_grid, log=False, grid_search=grid_search,
    custom_params_RF=custom_params_RF)
        pred_stake = predictions_XG / (1000)

        var_xg_missing = pd.DataFrame(
            {'time': list(range(time['test'].min(), time['test'].max() + 1))})
        var_xg_missing_inc = pd.DataFrame({
            'time': time['test'],
            'pred_xgb': pred_stake
        })
        var_xg_missing = pd.merge(var_xg_missing,
                                  var_xg_missing_inc,
                                  on='time',
                                  how='left')
        var_xg_missing['is_extr_t2m'] = [
            climate_year_is_extreme(inputDF, target, year, variable='t2m_mean')
            for year in var_xg_missing.time
        ]
        var_xg_missing['is_extr_tp'] = [
            climate_year_is_extreme(inputDF, target, year, variable='tp_tot')
            for year in var_xg_missing.time
        ]

        # save to pickle:
        name = f"var_{stake}.pkl"
        path = path_save_xgboost_stakes + "gap_filling/"
        createPath(path)
        with open(path + name, "wb") as fp:
            pickle.dump(var_xg_missing, fp)
            print(f"dictionary {name} saved successfully to {path}")

        extrm_years_t2m = [
            year for year in test_index if climate_year_is_extreme(
                inputDF, target, year, variable='t2m_mean')
        ]

        extrm_years_tp = [
            year for year in test_index if climate_year_is_extreme(
                inputDF, target, year, variable='tp_tot')
        ]
        return extrm_years_t2m, extrm_years_tp
    else:
        return [], []


def climate_year_is_extreme(inputDF, target, year, variable='t2m_mean'):
    inputDF_obs = inputDF.loc[target.index]
    perc_up = np.percentile(inputDF_obs[variable].values, 99)
    perc_low = np.percentile(inputDF_obs[variable].values, 1)
    var_year = inputDF[variable].loc[year]
    std = 0.5 * abs(np.std(inputDF_obs[variable].values))
    if var_year <= perc_up + std and var_year >= perc_low - std:
        return False
    else:
        return True
