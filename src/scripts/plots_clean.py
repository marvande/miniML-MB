from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re
import re
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter
from cmcrameri import cm

from scripts.xgb_metrics import *
from scripts.xgb_helpers import *
from scripts.PDD_helpers import *
from scripts.xgb_extreme import *

# ------------------ PLOTS FOR METRICS ------------------ #


def plot_losses_ann_monthly(df_metrics_annual,
                            df_metrics_monthly,
                            palette_grays,
                            color_palette,
                            model='lasso'):
    """
    Plot the losses for annual and monthly MeteoSuisse data.

    Parameters:
    - df_metrics_annual (DataFrame): DataFrame containing the metrics for annual data.
    - df_metrics_monthly (DataFrame): DataFrame containing the metrics for monthly data.
    - palette_grays (list): List of colors for the annual data plot.
    - color_palette (list): List of colors for the monthly data plot.
    - model (str): Model name (default: 'lasso').

    Returns:
    - None
    """
    fig = plt.figure(figsize=(20, 15))
    N, M = 4, 3
    ax1 = plt.subplot(N, M, 1)
    ax2 = plt.subplot(N, M, 2, sharey=ax1)
    ax3 = plt.subplot(N, M, 3, sharey=ax1)
    ax4 = plt.subplot(N, M, 4)
    ax5 = plt.subplot(N, M, 5, sharey=ax4)
    ax6 = plt.subplot(N, M, 6, sharey=ax4)

    PlotTrainValTestMet(df_metrics_annual,
                        ax1,
                        ax2,
                        ax3,
                        palette=color_palette,
                        model=model)
    ax1.set_ylabel('Annual MeteoSuisse')

    PlotTrainValTestMet(df_metrics_monthly,
                        ax4,
                        ax5,
                        ax6,
                        palette=color_palette,
                        model=model)
    PlotTrainValTestMet(df_metrics_annual,
                        ax4,
                        ax5,
                        ax6,
                        palette=palette_grays,
                        alpha=0.8,
                        model=model)
    ax4.set_ylabel('Monthly MeteoSuisse')

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
        ax.legend([], [], frameon=False)
    plt.tight_layout()


def PlotTrainValTestMet(df_metrics,
                        ax1,
                        ax2,
                        ax3,
                        palette,
                        alpha=1,
                        model='xgb'):
    """
    Plots the training, validation, and testing RMSE metrics for a given model.
    Args:
        df_metrics (DataFrame): The dataframe containing the metrics data.
        ax1 (AxesSubplot): The subplot for the training RMSE plot.
        ax2 (AxesSubplot): The subplot for the validation RMSE plot.
        ax3 (AxesSubplot): The subplot for the testing RMSE plot.
        palette (str or sequence): The color palette to use for the plots.
        alpha (float, optional): The transparency of the plot markers. Defaults to 1.
        model (str, optional): The name of the model. Defaults to 'xgb'.
    """
    df_metrics_grouped = df_metrics.groupby(
        ['glaciers', 'stakes',
         'stakes_full']).mean().reset_index().sort_values(by='stakes_full')

    evalPlot(df_metrics_grouped,
             ax1,
             metric=f'rmse_train_{model}',
             color_palette=palette,
             frequency=None,
             alpha=alpha,
             type_pred="")
    ax1.set_title('Training RMSE')

    evalPlot(df_metrics_grouped,
             ax2,
             metric=f'rmse_val_{model}',
             color_palette=palette,
             frequency=None,
             alpha=alpha,
             type_pred="")
    ax2.set_title('Validation RMSE')

    evalPlot(df_metrics_grouped,
             ax3,
             metric=f'rmse_{model}',
             color_palette=palette,
             frequency=None,
             alpha=alpha,
             type_pred="")
    ax3.set_title('Testing RMSE')


def evalPlot(
    df_metrics,
    ax,
    metric="rmse",
    add_std=True,
    color_palette=sns.color_palette("hls", 10),
    frequency="annual",
    type_pred="pred_XG",
    alpha=1,
    type_='scatter',
):
    """
    Plot evaluation metrics per stake.

    Parameters:
    - df_metrics (DataFrame): The dataframe containing the evaluation metrics.
    - ax (Axes): The matplotlib axes object to plot the data on.
    - metric (str, optional): The evaluation metric to plot. Default is "rmse".
    - add_std (bool, optional): Whether to add error bars representing the standard deviation. Default is True.
    - color_palette (list, optional): The color palette to use for plotting. Default is seaborn's "hls" palette.
    - frequency (str, optional): The frequency of the predictors. Default is "annual".
    - type_pred (str, optional): The type of prediction. Default is "pred_XG".
    - alpha (float, optional): The transparency of the plotted points. Default is 1.

    Returns:
    - None
    """
    long_name, y_label = get_long_name_metric(metric)

    if type_pred == "winter_pred_PDD":
        suffix = "_w"

    elif type_pred == "annual_pred_PDD":
        suffix = "_a"
    else:
        suffix = ""

    x_val = "stakes_full"
    metric = metric + suffix

    if type_ == 'scatter':
        g = sns.scatterplot(
            df_metrics,
            x=x_val,
            y=metric,
            ax=ax,
            size="num_points",
            hue="glaciers",
            # order=rmse_mean.index,
            palette=color_palette,
            color=(0.4, 0.6, 0.8, 0.5),
            alpha=alpha,
        )
    else:
        g = sns.barplot(
            df_metrics,
            x=x_val,
            y=metric,
            ax=ax,
            hue="glaciers",
            # order=rmse_mean.index,
            palette=color_palette,
            dodge=False,
            alpha=alpha,
        )

    # annotate:
    if x_val == "glaciers":
        for i, txt in enumerate(df_metrics["stakes"]):
            ax.annotate(
                txt,
                (
                    df_metrics[x_val][i],
                    df_metrics[metric][i] + 0.01 * df_metrics[metric][i],
                ),
            )
    if add_std:
        for i, txt in enumerate(df_metrics["stakes"]):
            ax.errorbar(
                df_metrics[x_val][i],
                df_metrics[metric][i],
                df_metrics[f"{metric}_std"][i],
                linestyle="None",
                color="grey",
                alpha=0.8,
                marker=None,
            )

    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(f"{y_label}")
    if frequency != None:
        ax.set_title(
            f"{long_name} per stake [{frequency} predictors]",
            fontsize=16,
        )
    else:
        ax.set_title(f"{long_name} per stake", fontsize=16)
    ax.set_xlabel("")
    if metric[:4] == "rmse" or metric[:3] == "mae":
        ax.axhline(0.9, linestyle="--", alpha=0.8, color="grey")
        ax.axhline(0.65, linestyle="--", alpha=0.8, color="grey")

    if metric[:5] == "nrmse":
        ax.axhline(1, linestyle="--", alpha=0.8, color="grey")
        ax.axhline(0.5, linestyle="--", alpha=0.8, color="grey")

    if metric[:11] == "correlation":
        ax.set_ylim(bottom=-1, top=1)
        ax.axhline(0, linestyle="--", alpha=0.8, color="grey")
        ax.axhline(0.6, linestyle="--", alpha=0.8, color="grey")
    if metric[:10] == "rsquared_2":
        ax.set_ylim(bottom=0, top=1)

    if (metric == "diff_rmse" or metric == "diff_correlation"
            or metric == "diff_mae" or metric == "diff_rsquared"):
        ax.axhline(0, linestyle="--", alpha=0.8, color="grey")
    h, l = g.get_legend_handles_labels()

    ax.legend(h,
              l,
              bbox_to_anchor=(1.05, 1),
              loc=2,
              ncol=2,
              borderaxespad=0.0,
              title='Glaciers')


def plotCompareFreqs(df_metrics_annual, df_metrics_half, df_metrics_seasonal,
                     df_metrics_monthly, metric, color_palette):
    """
    Plot and compare metrics for different time frequencies of climate data.

    Parameters:
    - df_metrics_annual: DataFrame containing annual metrics data
    - df_metrics_half: DataFrame containing half-yearly metrics data
    - df_metrics_seasonal: DataFrame containing seasonal metrics data
    - df_metrics_monthly: DataFrame containing monthly metrics data
    - metric: The metric to be plotted and compared

    Returns:
    - None
    """

    fig = plt.figure(figsize=(18, 10))
    M, N = 2, 2
    palette_grays = sns.color_palette(["#D0D5DF"])

    ax1 = plt.subplot(M, N, 1)
    ax2 = plt.subplot(M, N, 2)
    ax3 = plt.subplot(M, N, 3)
    ax4 = plt.subplot(M, N, 4)
    axes = [ax1, ax2, ax3, ax4]

    dic_freq = {
        'annual': df_metrics_annual,
        'half year': df_metrics_half,
        'seasonal': df_metrics_seasonal,
        'monthly': df_metrics_monthly
    }
    for i, key in enumerate(dic_freq.keys()):
        evalPlot(dic_freq[key],
                 axes[i],
                 metric=metric,
                 color_palette=color_palette,
                 frequency=key)
        if key == 'monthly':
            evalPlot(df_metrics_annual,
                     axes[i],
                     metric=metric,
                     color_palette=palette_grays,
                     add_std=False,
                     frequency='annual')
    for ax in [ax1, ax2, ax3, ax4]:
        ax.legend([], [], frameon=False)
    plt.tight_layout()


def ViolinPlotAnnToMonth(df_metrics_annual,
                         df_metrics_half,
                         df_metrics_seasonal,
                         df_metrics_monthly,
                         model='xgb'):
    """
    Generate a violin plot and scatter plot to visualize the mean RMSE values for different frequencies.

    Parameters:
    - df_metrics_annual: DataFrame containing metrics for the annual frequency.
    - df_metrics_half: DataFrame containing metrics for the half-yearly frequency.
    - df_metrics_seasonal: DataFrame containing metrics for the seasonal frequency.
    - df_metrics_monthly: DataFrame containing metrics for the monthly frequency.
    - model: Optional. The name of the model. Default is 'xgb'.

    Returns:
    - None
    """
    cmap = cm.batlow
    colors = get_cmap_hex(cmap, 4)
    color_palette = sns.color_palette(colors)
    dic_freq = {
        'annual': df_metrics_annual,
        'half-y': df_metrics_half,
        'seasonal': df_metrics_seasonal,
        'monthly': df_metrics_monthly
    }
    rmse, mae, freq_t = [], [], []
    for key in dic_freq.keys():
        rmse.append(dic_freq[key][f'rmse_{model}'])
        mae.append(dic_freq[key][f'mae_{model}'])
        freq_t.append(np.tile(key, len(dic_freq[key])))

    df_freq = pd.DataFrame({
        f'rmse_{model}': np.concatenate(rmse),
        f'mae_{model}': np.concatenate(mae),
        'freq': np.concatenate(freq_t),
        'stakes': np.tile(df_metrics_annual.stakes, 4),
        'glacier': np.tile(df_metrics_annual.glaciers, 4)
    })

    fig = plt.figure(figsize=(10, 3))
    ax = plt.subplot(1, 2, 1)
    sns.violinplot(df_freq,
                   x='freq',
                   y=f'rmse_{model}',
                   ax=ax,
                   palette=color_palette)
    ax.set_ylabel('Mean RMSE')
    ax.set_xlabel('Frequency')


# ------------------ PLOTS FOR COMPARING MODELS ------------------ #


def PlotCompare2Stakes(var_xg_monthly, metrics_monthly, stakes, kfold):
    colors = np.tile("#8CA6D9", len(var_xg_monthly.keys()))
    palette_grays = sns.color_palette(colors)

    fig = plt.figure(figsize=(15, 6))
    M, N = 2, 4

    for i, stake in enumerate(stakes):
        ax1 = plt.subplot(M, N, (i * 2) + 1)
        ax2 = plt.subplot(M, N, (i * 2) + 2)
        plotSingleStake(stake,
                        var_xg_monthly,
                        metrics_monthly,
                        ax1,
                        ax2,
                        freq="",
                        color="steelblue",
                        scaled=True,
                        kfold=kfold)

    plt.tight_layout()


def PlotDiffPDD(df_metrics_plot, color_palette, metric='rmse'):
    """
    Plot the difference between various RMSE metrics and RMSE(TIM) per stake.

    Parameters:
    - df_metrics_plot (DataFrame): The DataFrame containing the metrics data.
    - color_palette (list): The color palette to use for the plots.

    Returns:
    - None
    """

    long_name, ylabel = get_long_name_metric(metric)

    fig = plt.figure(figsize=(20, 5))
    M, N = 1, 3
    # Stripplot metric
    ax1 = plt.subplot(M, N, 1)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax1,
             metric=f'diff_{metric}_xgb',
             add_std=False,
             color_palette=color_palette,
             frequency=None,
             type_='bar')
    ax1.set_title(f'{long_name}(XGBoost)-{long_name}(TIM) per stake')

    ax3 = plt.subplot(M, N, 2, sharey=ax1)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax3,
             metric=f'diff_{metric}_lasso',
             add_std=False,
             color_palette=color_palette,
             frequency=None,
             type_='bar')
    ax3.set_title(f'{long_name}(Lasso)-{long_name}(TIM) per stake')

    ax2 = plt.subplot(M, N, 3, sharey=ax1)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax2,
             metric=f'diff_{metric}_xgb_bestvar',
             add_std=False,
             color_palette=color_palette,
             frequency=None,
             type_='bar')
    ax2.set_title(f'{long_name}(XGBoost++)-{long_name}(TIM) per stake')

    if metric == 'rmse':
        for ax in [ax1, ax2, ax3]:
            ax.axhline(200, linestyle="--", alpha=0.8, color="grey")

    for ax in [ax1, ax3]:
        ax.legend([], [], frameon=False)

    for ax in [ax2, ax3]:
        ax.set_ylabel('')

    plt.tight_layout()


def PlotCompareModelsPDD(df_metrics_plot,
                         color_palette,
                         palette_grays,
                         metric='rmse'):
    """
    Plots and compares different models against PDD (Temperature index model).

    Parameters:
    - df_metrics_plot (DataFrame): The dataframe containing the metrics for each model.
    - color_palette (list): The color palette to use for the models.
    - palette_grays (list): The grayscale palette to use for the PDD.

    Returns:
    - None
    """

    fig = plt.figure(figsize=(18, 5))
    M, N = 1, 3

    ax1 = plt.subplot(M, N, 1)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax1,
             metric=f'{metric}_pdd_a',
             color_palette=palette_grays,
             alpha=0.8,
             frequency=None,
             add_std=False)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax1,
             metric=f'{metric}_xgb',
             color_palette=color_palette,
             frequency=None,
             add_std=False)
    ax1.set_title('XGBoost versus TIM (in gray)')

    ax2 = plt.subplot(M, N, 2, sharey=ax1)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax2,
             metric=f'{metric}_pdd_a',
             color_palette=palette_grays,
             alpha=0.8,
             frequency=None,
             add_std=False)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax2,
             metric=f'{metric}_lasso',
             color_palette=color_palette,
             frequency=None,
             add_std=False)
    ax2.set_title('Lasso versus TIM (in gray)')

    ax3 = plt.subplot(M, N, 3, sharey=ax1)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax3,
             metric=f'{metric}_xgb_bestvar',
             color_palette=color_palette,
             frequency=None,
             add_std=False)
    evalPlot(df_metrics_plot.sort_values(by='stakes_full'),
             ax3,
             metric=f'{metric}_pdd_a',
             color_palette=palette_grays,
             alpha=0.8,
             frequency=None,
             add_std=False)
    ax3.set_title('XGBoost++ versus TIM (in gray)')

    for ax in [ax1, ax2, ax3]:
        ax.legend([], [], frameon=False)
    plt.tight_layout()


def PlotCompareXGBoostplus(
    stake,
    stake_full,
    var_pdd,
    metrics_pdd,
    df_metrics_pdd,
    var_xg_monthly,
    metrics_monthly,
    df_metrics_monthly,
    kfold,
    color_xgb,
    color_tim,
    marker_xgb,
    marker_tim,
    p1=0.02,
    p2=0.02,
    type_pred="pred_XG",
    label_xgb='XGB',
    add_pdd=True,
):
    """
    Plots and compares the results of XGBoost and PDD models.

    Args:
        stake (str): The stake value.
        var_pdd (dict): Dictionary containing PDD variables.
        metrics_pdd (dict): Dictionary containing PDD metrics.
        var_xg_monthly (dict): Dictionary containing XGBoost monthly variables.
        metrics_monthly (dict): Dictionary containing XGBoost monthly metrics.
        feature_list (list): List of feature names.
        kfold (int): Number of folds for cross-validation.
        p1 (float, optional): The x-coordinate for the legend text. Defaults to 0.02.
        p2 (float, optional): The y-coordinate for the legend text. Defaults to 0.02.
        type_pred (str, optional): The type of prediction. Defaults to "pred_XG".

    Returns:
        None
    """
    # color_palette = sns.color_palette("husl", len(MONTH_VAL.keys()))
    colors_cbfriendly = get_cmap_hex(cm.batlow, len(MONTH_VAL.keys()))
    palette = {}
    for ind in MONTH_VAL.keys():
        palette[MONTH_VAL[ind]] = colors_cbfriendly[ind - 1]
    palette_grays = sns.color_palette(np.tile("#8CA6D9", len(var_pdd.keys())))

    f, (ax1, ax2) = plt.subplots(1,
                                 2,
                                 figsize=(15, 3),
                                 gridspec_kw={"width_ratios": [1, 4]})

    if add_pdd:
        legend_pdd = plotSingleStake(stake,
                                     var_pdd,
                                     metrics_pdd,
                                     ax1,
                                     ax2,
                                     kfold=kfold,
                                     freq="",
                                     type_pred="annual_pred_PDD",
                                     color=color_tim,
                                     legend=False,
                                     label="PDD",
                                     marker=marker_tim)
    legend_xgb = plotSingleStake(stake,
                                 var_xg_monthly,
                                 metrics_monthly,
                                 ax1,
                                 ax2,
                                 kfold=kfold,
                                 freq="",
                                 color=color_xgb,
                                 legend=False,
                                 label=label_xgb,
                                 type_pred=type_pred,
                                 marker=marker_xgb)

    # ax2.set_ylim(-3,3)

    # add legend:
    df_stake = df_metrics_monthly[df_metrics_monthly['stakes_full'].apply(
        lambda x: x == stake_full)]
    df_pdd = df_metrics_pdd[df_metrics_pdd['stakes_full'].apply(
        lambda x: x == stake_full)]
    mae_xgb = df_stake[f"mae_xgb_full"]
    pearson_xgb = df_stake[f"corr_xgb_full"]
    pearson_pdd = df_pdd[f"corr_pdd_a_full"]
    mae_pdd = df_pdd[f"mae_pdd_a_full"]

    legend_text = "\n".join((
        r"$\mathrm{MAE}_{miniML}=%.3f, \mathrm{\rho}_{miniML}=%.2f$" % (
            mae_xgb,
            pearson_xgb,
        ),
        (r"$\mathrm{MAE}_{PDD}=%.3f,\mathrm{\rho}_{PDD}=%.2f$" % (
            mae_pdd,
            pearson_pdd,
        )),
    ))
    ax2.text(p1,
             p2,
             legend_text,
             transform=ax2.transAxes,
             verticalalignment="bottom",
             fontsize=16)

    ax2.tick_params(axis='both', labelsize=20)
    ax1.tick_params(axis='both', labelsize=20)

    plt.tight_layout()


# ------------------ PLOTS FOR MODEL PREDICTIONS ------------------ #


def plotSingleStakeLine(stake,
                        var_XG,
                        metrics,
                        ax1,
                        color,
                        kfold,
                        freq="annual",
                        scaled=False,
                        type_pred="pred_XG",
                        legend=True,
                        label="xgb",
                        marker='o'):
    pred_XG_stake = var_XG[type_pred]
    feat_test_stake = var_XG["feat_test"]
    feat_train_stake = var_XG["feat_train"]
    fold_ids_stake = var_XG["fold_id"]

    if type_pred == "winter_pred_PDD":
        suffix = "_w"

    elif type_pred == "annual_pred_PDD":
        suffix = "_a"
    else:
        suffix = ""

    rmse_stake = metrics["rmse" + suffix]
    mae_stake = metrics["mae" + suffix]
    pearson_stake = metrics["correlation" + suffix]

    stakeName = stake.split('-')[1]
    fold_ids = fold_ids_stake[stake]

    rmse = np.mean(rmse_stake[stake])
    mae = np.mean(mae_stake[stake])
    pearson = np.mean(pearson_stake[stake])

    gl = stake.split('-')[0]

    pred = pred_XG_stake[stake]
    truth = feat_test_stake[stake]["target" + suffix]

    if scaled:
        truth_scaled = np.concatenate(
            minMaxScaler(var_XG, truth.reshape(-1, 1)))
        stats_target = get_target_stats(truth_scaled, type_pred=type_pred)

    else:
        stats_target = get_target_stats(var_XG["feat_test"],
                                        type_pred=type_pred)

    nrmse = rmse / (np.abs(stats_target["mean"]))

    legend_text = linePlot(ax1,
                           feat_test_stake[stake],
                           feat_train_stake[stake],
                           pred_XG_stake[stake],
                           fold_ids,
                           rmse,
                           nrmse,
                           mae,
                           pearson,
                           kfold=kfold,
                           title=gl + "-" + stakeName + freq,
                           color=color,
                           type_pred=type_pred,
                           legend=legend,
                           marker=marker)

    return legend_text


def plotSingleStake(stake,
                    var_XG,
                    metrics,
                    ax1,
                    ax2,
                    color,
                    marker,
                    kfold,
                    freq="annual",
                    add_reg=False,
                    scaled=False,
                    type_pred="pred_XG",
                    legend=True,
                    label="xgb"):
    """
    Plots predictions versus target for a given stake.

    Args:
        stake (str): The stake name.
        var_XG (dict): Dictionary containing various variables and predictions.
        metrics (dict): Dictionary containing various metrics.
        color_palette (list): List of colors for plotting.
        ax1 (matplotlib.axes.Axes): The first subplot for the stake plot.
        ax2 (matplotlib.axes.Axes): The second subplot for the line plot.
        color (str): The color for the plot.
        kfold (int): The number of folds for cross-validation.
        freq (str, optional): The frequency of the data. Defaults to "annual".
        add_reg (bool, optional): Whether to add a regression line to the stake plot. Defaults to False.
        scaled (bool, optional): Whether the data is scaled. Defaults to False.
        type_pred (str, optional): The type of prediction. Defaults to "pred_XG".
        legend (bool, optional): Whether to show the legend in the line plot. Defaults to True.
        label (str, optional): The label for the line plot. Defaults to "xgb".
        
    Returns:
        str: The legend text for the line plot.
    """
    pred_XG_stake = var_XG[type_pred]
    feat_test_stake = var_XG["feat_test"]
    feat_train_stake = var_XG["feat_train"]
    fold_ids_stake = var_XG["fold_id"]

    if type_pred == "winter_pred_PDD":
        suffix = "_w"
    elif type_pred == "annual_pred_PDD":
        suffix = "_a"
    else:
        suffix = ""

    rmse_stake = metrics["rmse" + suffix]
    mae_stake = metrics["mae" + suffix]
    pearson_stake = metrics["correlation" + suffix]

    stakeName = stake.split('-')[1]
    fold_ids = fold_ids_stake[stake]

    rmse = np.mean(rmse_stake[stake])
    mae = np.mean(mae_stake[stake])
    pearson = np.mean(pearson_stake[stake])

    gl = stake.split('-')[0]

    pred = pred_XG_stake[stake]
    truth = feat_test_stake[stake]["target" + suffix]

    if scaled:
        truth_scaled = np.concatenate(
            minMaxScaler(var_XG, truth.reshape(-1, 1)))
        pred_scaled = np.concatenate(minMaxScaler(var_XG, pred.reshape(-1, 1)))

        stats_target = get_target_stats(truth_scaled, type_pred=type_pred)
        xlabel = "Scaled target"
        ylabel = "Scaled pred"
    else:
        stats_target = get_target_stats(var_XG["feat_test"],
                                        type_pred=type_pred)

        truth_scaled = truth / (1000)
        pred_scaled = pred / (1000)
        xlabel = "Observed PMB [m w.e.]"
        ylabel = "Predicted PMB"

    nrmse = rmse / (np.abs(stats_target["mean"]))

    predVsTruth(ax1,
                truth_scaled,
                pred_scaled,
                fold_ids,
                title=gl + "-" + stakeName + freq,
                color=color,
                add_reg=add_reg,
                xlabel=xlabel,
                ylabel=ylabel,
                label=label,
                marker=marker,
                alpha=0.8)

    legend_text = linePlot(ax2,
                           feat_test_stake[stake],
                           feat_train_stake[stake],
                           pred_XG_stake[stake],
                           fold_ids,
                           rmse,
                           nrmse,
                           mae,
                           pearson,
                           kfold=kfold,
                           title=gl + "-" + stakeName + freq,
                           color=color,
                           type_pred=type_pred,
                           legend=legend,
                           marker=marker)
    ax2.set_ylabel('[m w.e.]', fontsize=20)
    ax1.set_title('')
    ax2.set_title('')
    return legend_text


def predVsTruth(ax,
                target_test,
                predictions_XG,
                fold_ids,
                title=None,
                color="steelblue",
                add_reg=False,
                xlabel="Observed PMB [m w.e.]",
                ylabel="Pred PMB [m w.e.]",
                label="xgb",
                custom_legend_text='',
                alpha=0.8,
                marker='o'):
    """
    Plot the predicted values against the true values.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot the data on.
    - target_test (array-like): The true target values.
    - predictions_XG (array-like): The predicted target values.
    - fold_ids (array-like): The fold IDs for each data point.
    - title (str, optional): The title of the plot. Defaults to None.
    - color (str, optional): The color of the scatter plot. Defaults to "steelblue".
    - add_reg (bool, optional): Whether to add a regression line. Defaults to False.
    - xlabel (str, optional): The label for the x-axis. Defaults to "Target".
    - ylabel (str, optional): The label for the y-axis. Defaults to "Pred".
    - label (str, optional): The label for the scatter plot. Defaults to "xgb".
    """
    df = pd.DataFrame(data={
        "target": target_test,
        "pred": predictions_XG,
        "fold_id": fold_ids
    })
    sns.scatterplot(df,
                    x="target",
                    y="pred",
                    color=color,
                    ax=ax,
                    label=label,
                    alpha=alpha,
                    marker=marker)
    reg = LinearRegression().fit(df["target"].values.reshape(-1, 1),
                                 df["pred"].values)
    intercept = reg.intercept_
    slope = reg.coef_[0]

    # regression line
    if add_reg:
        pt = (0, intercept)
        ax.axline(pt, slope=slope, color="red", linestyle="-", linewidth=0.2)
        legend_text = "\n".join((r"$y = %.1fx +(%.1f)$" % (
            slope,
            intercept,
        ), ))
        ax.text(0.02,
                0.02,
                legend_text,
                transform=ax.transAxes,
                verticalalignment="bottom")

    if len(custom_legend_text) > 0:
        ax.text(0.03,
                0.98,
                custom_legend_text,
                transform=ax.transAxes,
                verticalalignment="top",
                fontsize=20)
        ax.legend([], [], frameon=False)

    # diagonal line
    pt = (0, 0)
    ax.axline(pt, slope=1, color="grey", linestyle="-", linewidth=0.2)
    ax.axvline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.axhline(0, color="grey", linestyle="-", linewidth=0.2)
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    # ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="both")
    ax.grid()
    ax.legend(fontsize=16)
    ax.set_title(title)


def linePlot(ax,
             feat_test,
             feat_train,
             predictions_XG,
             fold_ids,
             rmse,
             nrmse,
             mae,
             pearson,
             color,
             legend,
             kfold,
             title=None,
             type_pred="pred_XG",
             marker='o'):
    """
    Generate a line plot with error bars and scatter points for prediction and truth.

    Parameters:
    - ax (matplotlib.axes.Axes): The axes object to plot on.
    - feat_test (dict): Dictionary containing test features.
    - feat_train (dict): Dictionary containing train features.
    - predictions_XG (numpy.ndarray): Array of XGBoost predictions.
    - fold_ids (numpy.ndarray): Array of fold IDs.
    - rmse (float): Root Mean Squared Error.
    - nrmse (float): Normalized Root Mean Squared Error.
    - mae (float): Mean Absolute Error.
    - pearson (float): Pearson correlation coefficient.
    - color_palette (seaborn.color_palette): Color palette for the plot.
    - color (str): Color for the error bars.
    - legend (bool): Whether to display the legend.
    - kfold (int): Number of folds for cross-validation.
    - title (str): Title of the plot.
    - type_pred (str): Type of prediction.
    
    Returns:
    - legend_text (str): Text for the legend.
    """

    if type_pred == "winter_pred_PDD":
        suffix = "_w"
    elif type_pred == "annual_pred_PDD":
        suffix = "_a"
    else:
        suffix = ""

    # metrics
    total_df = assemblePredDf(
        feat_test["time"],
        predictions_XG,
        fold_ids,
        feat_test["target" + suffix],
        feat_train["time"][:int(len(feat_train["time"]))],
        feat_train["target" + suffix][:int(len(feat_train["time"]))],
        kfold=kfold,
    )
    total_df['truth'] = total_df['truth'] / (1000)
    total_df['pred'] = total_df['pred'] / (1000)
    total_df['error'] = total_df['error'] / (1000)

    # XGBoost predictions:
    plotErrorBars(total_df, ax, color, marker)
    sns.scatterplot(
        data=total_df,
        x="time",
        y="truth",
        marker=".",
        color="grey",
        alpha=0.8,
        ax=ax,
        s=5,
    )
    legend_text = "\n".join((
        r"$\mathrm{RMSE}=%.1f, \mathrm{\rho}=%.2f$" % (
            rmse,
            pearson,
        ),
        (r"$\mathrm{MAE}=%.2f,\mathrm{NRMSE}=%.2f$" % (
            mae,
            nrmse,
        )),
    ))
    if legend:
        ax.text(0.02,
                0.02,
                legend_text,
                transform=ax.transAxes,
                verticalalignment="bottom")
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("[mm w.e.]")
    # set y lim to min+10%:
    miny = total_df["truth"].min()
    if miny < 0:
        if miny < 0:
            if miny > -2:
                ax.set_ylim(bottom=2.5 * total_df["truth"].min())
            elif miny > -5 and miny < -2:
                ax.set_ylim(bottom=1.8 * total_df["truth"].min())
            else:
                ax.set_ylim(bottom=1.1 * total_df["truth"].min())
    else:
        ax.set_ylim(bottom=0)
    return legend_text


def plotErrorBars(total_df, ax, color, marker):
    """
    Plot error bars on a given axis.

    Args:
        total_df (DataFrame): The DataFrame containing the data.
        ax (Axes): The axis on which to plot the error bars.
        color (str): The color of the error bars.

    Returns:
        None
    """
    for fold_id in total_df.fold_ids.unique():
        sub_df = total_df.copy()
        mask = [sub_df.fold_ids == fold_id][0].values
        error_folds = []
        for i in range(len(sub_df["error"])):
            if mask[i]:
                error_folds.append(sub_df["error"].iloc[i])
            else:
                error_folds.append(np.nan)

        sub_df["error"] = error_folds

        lowlims = sub_df["error"] > 0
        uplims = sub_df["error"] <= 0

        plotline1, caplines1, barlinecols1 = ax.errorbar(
            x=sub_df["time"],
            y=sub_df["truth"],
            yerr=sub_df["error"].abs(),
            uplims=uplims,
            lolims=lowlims,
            linewidth=0.5,
            linestyle="-",
            alpha=1,
            color="grey",
            elinewidth=1,
            ecolor=color,
        )

        caplines1[0].set_marker(marker)
        caplines1[0].set_markersize(6)
        barlinecols1[0].set_linestyle("--")
        if len(caplines1) > 1:
            caplines1[1].set_marker(marker)
            caplines1[1].set_markersize(6)


# ------------------ PLOTS FOR FEATURE IMPORTANCES ------------------ #
def plotFIBest(df_total_annual,
               var_xg_monthly,
               feature_list,
               metric='rmse_xgb',
               title='XGB trained on annual MB',
               freq='monthly',
               type='xgboost'):

    color_palette = sns.color_palette("husl", len(MONTH_VAL.keys()))
    palette = {}
    for ind in MONTH_VAL.keys():
        palette[MONTH_VAL[ind]] = color_palette[ind - 1]

    fig = plt.figure(figsize=(15, 4))
    ax = plt.subplot(1, 3, 2)
    mod_stakesdf = df_total_annual[df_total_annual[metric] < 900]
    good_stakes = [
        mod_stakesdf.iloc[i]['glaciers'] + '_' + mod_stakesdf.iloc[i]['stakes']
        for i in range(len(mod_stakesdf))
    ]

    fidf = pd.DataFrame()
    for stake in good_stakes:
        fidf = pd.concat(
            [fidf, pd.DataFrame(var_xg_monthly['all_fi'][stake])], axis=0)
    fidf.columns = feature_list
    feature_import = fidf.transpose().mean(axis=1)
    FIPlot(
        ax,
        feature_list,
        feature_import,
        title=
        f'Average FI for ({len(mod_stakesdf)}) stakes with {re.split("_", metric)[0]} < 900 mm w.e.',
        palette=palette,
        freq=freq,
        type=type)

    ax = plt.subplot(1, 3, 3)
    good_stakesdf = df_total_annual[df_total_annual[metric] < 650]
    good_stakes = [
        good_stakesdf.iloc[i]['glaciers'] + '_' +
        good_stakesdf.iloc[i]['stakes'] for i in range(len(good_stakesdf))
    ]

    fidf = pd.DataFrame()
    for stake in good_stakes:
        fidf = pd.concat(
            [fidf, pd.DataFrame(var_xg_monthly['all_fi'][stake])], axis=0)
    fidf.columns = feature_list
    feature_import = fidf.transpose().mean(axis=1)
    FIPlot(
        ax,
        feature_list,
        feature_import,
        title=
        f'Average FI for ({len(good_stakesdf)}) stakes with {re.split("_", metric)[0]} < 650 mm w.e.',
        palette=palette,
        freq=freq,
        type=type)

    # all stakes
    ax = plt.subplot(1, 3, 1)
    all_stakes = [
        df_total_annual.iloc[i]['glaciers'] + '_' +
        df_total_annual.iloc[i]['stakes'] for i in range(len(df_total_annual))
    ]

    fidf = pd.DataFrame()
    for stake in all_stakes:
        fidf = pd.concat(
            [fidf, pd.DataFrame(var_xg_monthly['all_fi'][stake])], axis=0)
    fidf.columns = feature_list
    feature_import = fidf.transpose().mean(axis=1)
    FIPlot(ax,
           feature_list,
           feature_import,
           title=f'Average FI for all ({len(df_total_annual)}) stakes',
           palette=palette,
           freq=freq,
           type=type)

    plt.suptitle(title)
    plt.tight_layout()


def FIPlot(ax,
           feature_list,
           feature_import,
           palette,
           title=None,
           freq='monthly',
           type='xgboost'):
    """
    Plot feature importances.

    Parameters:
    - ax: The matplotlib axes object to plot on.
    - feature_list: A list of feature names.
    - feature_import: A list of feature importances.
    - palette: The color palette to use for the plot.
    - title: The title of the plot (optional).
    - freq: The frequency of the data (monthly, annual, seasonal, halfy).
    - type: The type of feature importances (xgboost or other).

    Returns:
    - None
    """

    feature_importdf = pd.DataFrame(data={
        "variables": feature_list,
        "feat_imp": feature_import
    })
    if type == 'xgboost':
        feature_importdf.sort_values(by="feat_imp",
                                     ascending=True,
                                     inplace=True)

    # Keep only feature more important than 2%
    feature_importdf = feature_importdf[feature_importdf.feat_imp > 0.02]
    if freq == 'monthly' or freq == 'seasonal':
        feature_importdf['freq'] = feature_importdf.variables.apply(
            lambda x: re.split('_', x)[1])
    elif freq == 'annual' or freq == 'halfy':
        feature_importdf['freq'] = feature_importdf.variables.apply(
            lambda x: re.split('_', x)[0])

    if freq == 'monthly':
        sns.barplot(feature_importdf,
                    x='feat_imp',
                    y='variables',
                    hue='freq',
                    dodge=False,
                    palette=palette,
                    ax=ax)
    else:
        cmap = cm.batlow
        colors = get_cmap_hex(cmap, feature_importdf.variables.nunique())
        sns.barplot(feature_importdf,
                    x='feat_imp',
                    y='variables',
                    hue='variables',
                    dodge=False,
                    palette=sns.color_palette(colors),
                    ax=ax)

    for bars in ax.containers:
        if type == 'xgboost':
            ax.bar_label(bars, fmt="{:.2%}")
        else:
            ax.bar_label(bars, fmt="{:.2}")

    if type == 'xgboost':
        ax.set_xlim(xmax=1)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
    ax.set_ylabel('Variables')
    ax.set_xlabel('Feature importance')
    ax.set_title(title)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


def PlotMultiFI(pos,
                df_total_annual,
                var_xg_monthly,
                metrics_monthly,
                feature_list,
                kfold,
                type_pred="pred_XG",
                color_palette=sns.color_palette("hls", 10)):
    M, N = 6, 2
    fig = plt.figure(figsize=(15, 18))
    i = 1
    for j, stake_full in enumerate(
            df_total_annual.stakes_full[(M) * (pos - 1):(M) * (pos)]):
        stake_pos = j + (pos - 1) * M
        stake = df_total_annual.iloc[stake_pos][
            'glaciers'] + '_' + df_total_annual.iloc[stake_pos]['stakes']
        ax1 = plt.subplot(M, N, i)
        plotFI_all(stake,
                   var_xg_monthly,
                   feature_list,
                   ax1,
                   type_pred=type_pred)
        ax1.set_title(f'{stake_full}: annual MB')
        ax1.legend([], frameon=False)

        ax3 = plt.subplot(M, N, i + 1)
        # single plot
        legend2 = plotSingleStakeLine(stake,
                                      var_xg_monthly,
                                      metrics_monthly,
                                      ax3,
                                      kfold=kfold,
                                      color="seagreen",
                                      freq="annual",
                                      scaled=False,
                                      type_pred=type_pred,
                                      legend=False)

        # add legend:
        rmse_xgb = np.mean(metrics_monthly["rmse"][stake])
        pearson_xgb = np.mean(metrics_monthly["correlation"][stake])

        legend_text = "\n".join(
            (r"$\mathrm{RMSE}_{xgb}=%.1f, \mathrm{\rho}_{xgb}=%.2f$" % (
                rmse_xgb,
                pearson_xgb,
            ), ))
        ax3.text(0.02,
                 0.02,
                 legend_text,
                 transform=ax3.transAxes,
                 verticalalignment="bottom")

        ax3.set_title(f'{stake_full}: annual MB')
        i += 2
    plt.tight_layout()


def plotFI_all(stake, var_xg_monthly, feature_list, ax, type_pred='pred_XG'):
    """
    Plot feature importance for all folds.

    Args:
        stake (str): The stake value.
        var_xg_monthly (dict): The dictionary containing the feature importance data.
        feature_list (list): The list of feature names.
        ax (matplotlib.axes.Axes): The axes object to plot the feature importance.
        type_pred (str, optional): The type of prediction. Defaults to 'pred_XG'.

    Returns:
        None
    """
    fidf = pd.DataFrame(var_xg_monthly['all_fi'][stake],
                        columns=feature_list,
                        index=[f'fold_{i}' for i in range(5)]).transpose()
    fidf['mean'] = fidf.mean(axis=1)

    col0 = fidf['fold_0']
    fold_id = [[0 for i in range(len(col0))]]
    for i, col in enumerate(fidf.columns[1:-1]):
        col0 = pd.concat([col0, fidf[col]], axis=0)
        fold_id.append([i + 1 for j in range(len(fidf[col]))])

    fidf_ex = pd.DataFrame(data={
        'fi': col0,
        'fold_id': np.concatenate(fold_id)
    })
    fidf_ex = fidf_ex.reset_index()

    # add mean:
    fidf_ex['mean_fi'] = np.tile(fidf['mean'], 5)

    if type_pred == 'pred_XG':
        # transform into percent
        fidf_ex['fi'] = fidf_ex['fi'] * 100
        fidf_ex['mean_fi'] = fidf_ex['mean_fi'] * 100
        # Keep only feature more important than 2%
        fidf_ex = fidf_ex[fidf_ex.mean_fi > 2]
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    sns.scatterplot(fidf_ex, x='index', y='fi', hue='fold_id', ax=ax)
    ax.tick_params(axis="x", rotation=90)


def plotDecadesFI(path_decades,
                  full_stakes,
                  feature_list,
                  N=1,
                  month_val=MONTH_VAL,
                  figsize=(10, 10)):
    fi_decades, rmse_decades = {}, {}
    for stakeName in full_stakes:
        with open(path_decades + f'var_{stakeName}_full.pkl', "rb") as fp:
            var_gl = pickle.load(fp)
        with open(path_decades + f'metrics_{stakeName}_full.pkl', "rb") as fp:
            metrics_gl = pickle.load(fp)
        years = list(var_gl['all_fi'].keys())
        for year in years:
            # Add mean FI over all 5 folds for decade
            updateDic(fi_decades, year, var_gl['mean_fi'][year])
            # Add mean RMSE over all 5 folds for decade
            updateDic(rmse_decades, year, np.mean(metrics_gl['rmse'][year]))

    mean_fi, rmse_s, years_s = [], [], []
    for year in years:
        # Mean over all stakes
        mean_fi.append(np.mean(fi_decades[year], axis=0))
        # Mean over all stakes
        rmse_s.append(np.mean(rmse_decades[year], axis=0))
        years_s.append(np.tile(year, 24))

    df = pd.DataFrame(
        data={
            'features': np.tile(feature_list, int(6 / N)),
            'FI': np.concatenate(mean_fi),
            'years': np.concatenate(years_s)
        })
    df['month'] = df.features.apply(lambda x: re.split('_', x)[1])

    color_palette = sns.color_palette("husl", len(month_val.keys()))
    palette = {}
    for ind in month_val.keys():
        palette[month_val[ind]] = color_palette[ind - 1]

    fig = plt.figure(figsize=figsize)
    g = sns.FacetGrid(df, col="years", col_wrap=3, height=4, sharey=False)
    g.map_dataframe(sns.barplot,
                    x='FI',
                    y='features',
                    hue='month',
                    dodge=False,
                    palette=palette)
    g.add_legend()

    def to_percent(y, position):
        s = str(round(100 * y))
        return s + '%'

    formatter = FuncFormatter(to_percent)
    plt.gca().xaxis.set_major_formatter(formatter)

    for i, ax in enumerate(g.axes.flat):
        rmse_year = rmse_s[i]
        legend_text = "\n".join(
            (r"$\mathrm{RMSE}_{xgb}=%.1f$" % (rmse_year, ), ))
        ax.text(0.5,
                0.01,
                legend_text,
                transform=ax.transAxes,
                verticalalignment="bottom")
    plt.tight_layout()


def plotDecades(path_decades, stakeName, var_xg_monthly, metrics_monthly,
                kfold, feature_list, palette):
    with open(path_decades + f'var_{stakeName}_full.pkl', "rb") as fp:
        var_gl = pickle.load(fp)

    fig = plt.figure(figsize=(15, 10))
    years = list(var_gl['all_fi'].keys())
    years.sort()

    # Add total FI:
    ax1 = plt.subplot(3, 3, 1)
    fidf = pd.DataFrame(var_xg_monthly['all_fi'][stakeName])
    fidf.columns = feature_list
    feature_import = fidf.transpose().mean(axis=1)
    FIPlot(ax1,
           feature_list,
           feature_import,
           title=f'Average FI trained on all decades',
           palette=palette)

    # colors:
    colors = np.tile("#8CA6D9", len(var_xg_monthly.keys()))
    palette_grays = sns.color_palette(colors)

    ax2 = plt.subplot(3, 3, 2)
    ax3 = plt.subplot(3, 3, 3)
    legend_xgb = plotSingleStake(stakeName,
                                 var_xg_monthly,
                                 metrics_monthly,
                                 ax2,
                                 ax3,
                                 kfold=kfold,
                                 freq="",
                                 type_pred="pred_XG",
                                 color=palette['June'],
                                 legend=False,
                                 label="XGB")
    # add legend:
    rmse_xgb = np.mean(metrics_monthly["rmse"][stakeName])
    pearson_xgb = np.mean(metrics_monthly["correlation"][stakeName])

    legend_text = "\n".join(
        (r"$\mathrm{RMSE}_{xgb}=%.1f, \mathrm{\rho}_{xgb}=%.2f$" % (
            rmse_xgb,
            pearson_xgb,
        ), ))
    p1 = 0.02
    p2 = 0.02
    ax3.text(p1,
             p2,
             legend_text,
             transform=ax3.transAxes,
             verticalalignment="bottom")
    ax2.set_title('Pred vs target point MB')
    ax3.set_title('Time series point MB')

    for i, year in enumerate(years):
        ax1 = plt.subplot(3, 3, i + 4)
        fidf = pd.DataFrame(var_gl['all_fi'][year])
        fidf.columns = feature_list
        feature_import = fidf.transpose().mean(axis=1)
        FIPlot(ax1,
               feature_list,
               feature_import,
               title=f'FI for decade: {year}',
               palette=palette)
    plt.suptitle(f'{stakeName}')
    plt.tight_layout()


# ------------------ PLOTS FOR HYPERPARAMETERS ------------------ #


def plotRF_HPrange(metrics_RF, param_grid_RF):
    """
   Plot the relationship between hyperparameters and RMSE for different stakes and glaciers.

   Args:
      metrics_RF (dict): A dictionary containing the metrics for different hyperparameters.
      param_grid_RF (dict): A dictionary containing the hyperparameter grid.

   Returns:
      None
   """
    mean_rmse = [
        np.mean(metrics_RF['rmse'][stake])
        for stake in metrics_RF['rmse'].keys()
    ]
    dfHP = pd.DataFrame(
        data={
            'hp_n_estimators': metrics_RF['hp_n_estimators'].values(),
            'hp_max_depth': metrics_RF['hp_max_depth'].values(),
            'rmse': mean_rmse,
            'stakes': metrics_RF['hp_n_estimators'].keys(),
        })
    dfHP['glaciers'] = dfHP['stakes'].apply(lambda x: re.split('_', x)[0])
    # Alpha
    fig = plt.figure(figsize=(8, 3))
    ax1 = plt.subplot(1, 2, 1)
    sns.stripplot(dfHP, x='glaciers', y='hp_n_estimators', ax=ax1, hue='rmse')
    ax1.tick_params(axis="x", rotation=90)
    ax1.set_xlabel('')
    ax1.axhline(param_grid_RF['n_estimators'][-1])
    ax1.legend([], frameon=False)

    ax1 = plt.subplot(1, 2, 2)
    sns.stripplot(dfHP, x='glaciers', y='hp_max_depth', ax=ax1, hue='rmse')
    ax1.tick_params(axis="x", rotation=90)
    ax1.axhline(param_grid_RF['max_depth'][-1])
    ax1.set_xlabel('')
    ax1.legend([], frameon=False)

    plt.tight_layout()


def plotHyperParam(metrics_annual, param_grid, ax1, ax2, ax3, freq='annual'):
    mean_rmse = [
        np.mean(metrics_annual['rmse'][stake])
        for stake in metrics_annual['rmse'].keys()
    ]
    dfHP = pd.DataFrame(
        data={
            'learning_rate': metrics_annual['hp_lr'].values(),
            'num_estimators': metrics_annual['hp_ne'].values(),
            'max_depth': metrics_annual['hp_md'].values(),
            'stakes': metrics_annual['hp_lr'].keys(),
            'rmse': mean_rmse
        })
    dfHP['glaciers'] = dfHP['stakes'].apply(lambda x: re.split('_', x)[0])

    # Learning rate
    sns.stripplot(dfHP, x='glaciers', y='learning_rate', ax=ax1, hue='rmse')
    ax1.axhline(param_grid['learning_rate'][0], linestyle='--', color='red')
    ax1.axhline(param_grid['learning_rate'][-1], linestyle='--', color='red')
    ax1.tick_params(axis="x", rotation=90)
    ax1.set_title(f'Frequency: {freq}')
    ax1.set_xlabel('')
    ax1.legend([], frameon=False)

    # Number of estimators
    sns.stripplot(dfHP, x='glaciers', y='num_estimators', ax=ax2, hue='rmse')
    ax2.axhline(param_grid['n_estimators'][0], linestyle='--', color='red')
    ax2.axhline(param_grid['n_estimators'][-1], linestyle='--', color='red')
    ax2.tick_params(axis="x", rotation=90)
    ax2.set_xlabel('')
    ax2.legend([], frameon=False)

    # Max depth
    sns.stripplot(dfHP,
                  x='glaciers',
                  y='max_depth',
                  ax=ax3,
                  hue='rmse',
                  jitter=True)
    ax3.axhline(param_grid['max_depth'][0], linestyle='--', color='red')
    ax3.axhline(param_grid['max_depth'][-1], linestyle='--', color='red')
    ax3.tick_params(axis="x", rotation=90)
    ax3.set_xlabel('')
    ax3.legend(bbox_to_anchor=(1, 1))


def paramPlot(metric,
              ax,
              df_params,
              df_params_avg,
              color_palette,
              type_="scatter"):
    x_val = "stakes_full"
    y_labels = {
        "c_prec": "Prec. facator",
        "DDFsnow": "DDF snow",
    }

    if type_ == "scatter":
        g = sns.scatterplot(
            df_params,
            x=x_val,
            y=metric,
            ax=ax,
            palette=["red", "grey"],
            color="grey",
            # alpha=0.3,
            hue="annual_match",
            marker="x",
        )
        g = sns.scatterplot(
            df_params_avg,
            x=x_val,
            y=metric,
            ax=ax,
            # hue="glaciers",
            palette=color_palette,
            color=(0.4, 0.6, 0.8, 0.5),
        )

    elif type_ == "boxplot":
        g = sns.boxplot(df_params,
                        x=x_val,
                        y=metric,
                        ax=ax,
                        palette=color_palette,
                        showmeans=True,
                        dodge=False,
                        meanprops={
                            'marker': '^',
                            'markerfacecolor': 'white'
                        })
    ax.tick_params(axis="x", rotation=90)
    ax.set_ylabel(f"{y_labels[metric]}")
    ax.set_xlabel("")

    return g


def plotParamRange(var_pdd, color_palette):
    df_params = (dfPDDParams(var_pdd).explode(
        column=["c_prec", "DDFsnow", "winter_match", "annual_match"
                ]).sort_values(by="stakes_full"))

    df_params_avg = df_params.groupby("stakes_full").mean()
    df_params_avg = df_params_avg.merge(
        dfPDDParams(var_pdd)[["glaciers", "stakes", "stakes_full"]],
        on="stakes_full")

    fig = plt.figure(figsize=(18, 10))

    ax = plt.subplot(2, 2, 1)
    g = paramPlot("c_prec", ax, df_params, df_params_avg, color_palette)
    ax.legend([], [], frameon=False)

    ax = plt.subplot(2, 2, 2)
    g = paramPlot("DDFsnow", ax, df_params, df_params_avg, color_palette)

    h, l = g.get_legend_handles_labels()

    ax.legend(
        h,
        l,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        ncol=2,
        borderaxespad=0.0,
    )

    ax = plt.subplot(2, 2, 3)
    g = paramPlot("c_prec",
                  ax,
                  df_params,
                  df_params_avg,
                  color_palette,
                  type_="boxplot")
    ax.legend([], [], frameon=False)

    ax = plt.subplot(2, 2, 4)
    g = paramPlot("DDFsnow",
                  ax,
                  df_params,
                  df_params_avg,
                  color_palette,
                  type_="boxplot")

    h, l = g.get_legend_handles_labels()

    ax.legend(
        h,
        l,
        bbox_to_anchor=(1.05, 1),
        loc=2,
        ncol=2,
        borderaxespad=0.0,
    )
    plt.tight_layout()


def PlotFeatClusters(df_clusters_per_feat):
    # colors = get_cmap_hex(cm.devon, 5)
    colors = ['#e3f1ff', '#f7ebf9', '#fff0f0', '#de77ae']
    num_c = df_clusters_per_feat[df_clusters_per_feat.feature ==
                                 't2m'].groupby('cluster').count()[0].values
    df = df_clusters_per_feat.groupby(['feature',
                                       'cluster']).mean().reset_index()
    freq_c, months, c, feat = [], [], [], []
    for i in range(len(df)):
        freq_c.append(df.iloc[i][[i for i in range(0, 12)]].values)
        months.append(list(MONTH_VAL.values()))
        c.append(np.tile(df.iloc[i]['cluster'], len(MONTH_VAL.values())))
        feat.append(np.tile(df.iloc[i]['feature'], len(MONTH_VAL.values())))

    df = pd.DataFrame(
        data={
            'freq_var': np.concatenate(freq_c),
            'month': np.concatenate(months),
            'cluster': np.concatenate(c),
            'feature': np.concatenate(feat)
        })

    g = sns.FacetGrid(
        df,
        row="feature",
        col='cluster',
        #hue='month',
        height=2.5,
        aspect=1.5)
    g.map(sns.barplot,
          "month",
          "freq_var",
          orient='v',
          color='#4d4d4d',
          order=INVERSE_MONTH_POS.keys())

    for col_val, ax in g.axes_dict.items():
        ax.set_ylabel('Frequ. of month')
        ax.set_xlabel('')
        ax.set_ylim(top=1)
        ax.tick_params(axis="x", rotation=90)
        legend_text = "\n".join((r"$N_s=%.1f$" % (num_c[col_val[1]], ), ))
        ax.text(0.02,
                0.8,
                legend_text,
                transform=ax.transAxes,
                verticalalignment="bottom",
                fontsize=16)
        ax.set_facecolor(colors[col_val[1]])
        ax.set_title('')
    g.add_legend()


def plotClusterStats(df_info, mean_df):
    # colors = ['#74add1', '#fee090', '#de77ae']
    # colors = get_cmap_hex(cm.devon, 5)
    colors = ['#82a7cc', '#d3bad8', '#ccafaf', '#de77ae']
    markers = ['o', 's', 'X']
    color_palette = sns.color_palette(colors)
    M, N = 1, 5
    fig = plt.figure(figsize=(18, 5))
    titlefont = 20
    ax1 = plt.subplot(M, N, 2)
    scatter_arguments = {
        's': 15,
        'alpha': 0.8,
        'palette': color_palette,
        'hue': 'cluster',
        'jitter': True,
    }
    for i, marker in enumerate(markers):
        sns.stripplot(
        df_info[df_info.cluster == i+1],
        x='cluster',
        y='elevation',
        ax=ax1,
        marker = marker,
        **scatter_arguments,
        )
    ax1.set_ylabel('[m]', fontsize=titlefont)
    ax1.set_title('Elevation', fontsize=titlefont)

    ax2 = plt.subplot(M, N, 3)
    df_info['training_mb'] = df_info['training_mb'] / 1000
    
    for i, marker in enumerate(markers):
        sns.stripplot(
        df_info[df_info.cluster == i+1],
        x='cluster',
        y='training_mb',
        ax=ax2,
        marker = marker,
        **scatter_arguments,
        )
    
    ax2.set_title('Mean PMB', fontsize=titlefont)
    ax2.set_ylabel('[m w.e.]', fontsize=titlefont)

    ax3 = plt.subplot(M, N, 1)
    
    for i, marker in enumerate(markers):
        sns.scatterplot(
        df_info[df_info.cluster == i+1],
        x='lon',
        y='lat',
        ax=ax3,
        marker = marker,
        palette=color_palette,
                    hue='cluster',
                    s = 200
        )


    ax3.set_title('Position', fontsize=titlefont)
    ax3.set_ylabel('Latitude [$\degree$]', fontsize=titlefont)
    ax3.set_xlabel('Longitude [$\degree$]', fontsize=titlefont)
    ax3.legend([])

    ax4 = plt.subplot(M, N, 4)
    
    for i, marker in enumerate(markers):
        sns.stripplot(
        df_info[df_info.cluster == i+1],
        x='cluster',
        y='training_time',
        ax=ax4,
        marker = marker,
        **scatter_arguments,
        )
    
    ax4.set_ylabel('', fontsize=titlefont)
    ax4.set_title('Average year', fontsize=titlefont)

    ax5 = plt.subplot(M, N, 5)    
    for i, marker in enumerate(markers):
        sns.stripplot(
        df_info[df_info.cluster == i+1],
        x='cluster',
        y='training_length',
        ax=ax5,
        marker = marker,
        **scatter_arguments,
        )
    
    ax5.set_ylabel('')
    ax5.set_title('Number of years', fontsize=titlefont)
    for ax in [ax1, ax2, ax4, ax5]:
        ax.legend([])
        ax.set_xlabel('Cluster', fontsize=titlefont)
    plt.tight_layout()


def assemblePredDf(time_test, pred, fold_ids, truth_test, time_train,
                   truth_train, kfold):
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


def plotAttrsStakes(var_xg_monthly, names_stakes, stakes_names, stakes_low):
    # Get attributes of clusters:
    cl_elev, cl_lat, cl_lon, stakes, glaciers, glshort = [], [], [], [], [], []
    training_mb, training_y, len_training = [], [], []
    is_outlier = [names_stakes[stake] in stakes_low for stake in stakes_names]
    for stake in stakes_names:
        newName = names_stakes[stake]
        f_stake = read_stake_csv(path_glacattr, f'{stake}_mb.csv')
        cl_elev.append(np.mean(f_stake.height))
        cl_lat.append(np.mean(f_stake.lat))
        cl_lon.append(np.mean(f_stake.lon))
        len_training.append(
            len(var_xg_monthly['feat_train'][newName]['target']) / 5)
        # mean training mb
        training_mb.append(
            np.mean(var_xg_monthly['feat_train'][newName]['target']))
        training_y.append(
            int(np.mean(var_xg_monthly['feat_train'][newName]['time'])))
        stakes.append(newName)
        glaciers.append(re.split('-', newName)[0])
        glshort.append(GL_SHORT[re.split('-', newName)[0]] + '-' +
                       re.split('-', newName)[1])

    df_info = pd.DataFrame({
        'elevation': cl_elev,
        'lon': cl_lon,
        'lat': cl_lat,
        'training_mb': training_mb,
        'training_time': training_y,
        'training_length': len_training,
        'stakes': stakes,
        'glaciers': glaciers,
        'stakesfull': glshort,
        'outlier': is_outlier
    })

    colors = ['#82a7cc', '#d3bad8', '#ccafaf', '#de77ae']
    color_palette = sns.color_palette(colors)
    M, N = 1, 5
    fig = plt.figure(figsize=(18, 5))
    titlefont = 20
    ax1 = plt.subplot(M, N, 2)
    scatter_arguments = {
        's': 15,
        'alpha': 0.8,
        'palette': color_palette,
        'hue': 'outlier',
        'jitter': True
    }
    sns.stripplot(
        df_info,
        x='glaciers',
        y='elevation',
        ax=ax1,
        **scatter_arguments,
    )
    ax1.set_ylabel('[m]', fontsize=titlefont)
    ax1.set_title('Elevation', fontsize=titlefont)

    ax2 = plt.subplot(M, N, 3)
    df_info['training_mb'] = df_info['training_mb'] / 1000
    sns.stripplot(
        df_info,
        x='glaciers',
        y='training_mb',
        ax=ax2,
        **scatter_arguments,
    )
    ax2.set_title('Mean PMB', fontsize=titlefont)
    ax2.set_ylabel('[m w.e.]', fontsize=titlefont)

    ax3 = plt.subplot(M, N, 1)
    sns.scatterplot(df_info,
                    x='lon',
                    y='lat',
                    ax=ax3,
                    palette=color_palette,
                    hue='outlier',
                    s=200)
    ax3.set_title('Position', fontsize=titlefont)
    ax3.set_ylabel('Latitude [$\degree$]', fontsize=titlefont)
    ax3.set_xlabel('Longitude [$\degree$]', fontsize=titlefont)
    ax3.legend([], [], frameon=False)

    ax4 = plt.subplot(M, N, 4)
    sns.stripplot(
        df_info,
        x='glaciers',
        y='training_time',
        ax=ax4,
        **scatter_arguments,
    )
    ax4.set_ylabel('', fontsize=titlefont)
    ax4.set_title('Average year', fontsize=titlefont)

    ax5 = plt.subplot(M, N, 5)
    g = sns.stripplot(
        df_info,
        x='glaciers',
        y='training_length',
        ax=ax5,
        **scatter_arguments,
    )
    h, l = g.get_legend_handles_labels()
    ax5.set_ylabel('')
    ax5.set_title('Number of years', fontsize=titlefont)
    ax5.set_xlabel('', fontsize=titlefont)
    ax5.tick_params(axis="x", rotation=90)
    ax5.legend(h,
               l,
               bbox_to_anchor=(1.05, 1),
               loc=2,
               ncol=1,
               borderaxespad=0.0,
               fontsize=16)
    for ax in [ax1, ax2, ax4]:
        ax.legend([], [], frameon=False)
        ax.set_xlabel('', fontsize=titlefont)
        ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()


def plotExtremePreds(stake,
                     stakeShort,
                     ax1,
                     ax2,
                     var_pdd_a,
                     metrics_pdd_a,
                     df_metrics_pdd,
                     var_xg_best,
                     metrics_best,
                     df_metrics_best,
                     var_xg_missing,
                     kfold,
                     color_tim,
                     color_xgbplus,
                     marker_tim,
                     marker_xgb,
                     connect=True):
    plotSingleStake(stake,
                    var_pdd_a,
                    metrics_pdd_a,
                    ax1,
                    ax2,
                    kfold=kfold,
                    freq="",
                    type_pred="annual_pred_PDD",
                    color=color_tim,
                    legend=False,
                    label="PDD",
                    marker=marker_tim)
    plotSingleStake(stake,
                    var_xg_best,
                    metrics_best,
                    ax1,
                    ax2,
                    kfold=kfold,
                    freq="",
                    color=color_xgbplus,
                    legend=False,
                    label='miniML',
                    type_pred="pred_XG",
                    marker=marker_xgb)
    sns.scatterplot(var_xg_missing,
                    x='time',
                    y='pred_xgb',
                    marker=marker_xgb,
                    hue='is_extr_t2m',
                    palette=sns.color_palette(['grey', '#FF0000']),
                    s=60,
                    ax=ax2)
    if connect:
        sns.lineplot(var_xg_missing,
                     x='time',
                     y='pred_xgb',
                     marker=True,
                     color='grey',
                     ax=ax2)
    ax2.legend(title='T is extreme',
               fontsize=15,
               loc='lower right',
               title_fontsize=15,
               ncols=2)

    # add legend:
    df_stake = df_metrics_best[df_metrics_best['stakes_full'].apply(
        lambda x: x == stakeShort)]
    df_pdd = df_metrics_pdd[df_metrics_pdd['stakes_full'].apply(
        lambda x: x == stakeShort)]
    mae_xgb = df_stake[f"mae_xgb_full"]
    pearson_xgb = df_stake[f"corr_xgb_full"]
    pearson_pdd = df_pdd[f"corr_pdd_a_full"]
    mae_pdd = df_pdd[f"mae_pdd_a_full"]

    std_ml = np.std(var_xg_best['pred_XG'][stake]) / (1000)
    std_pdd = np.std(var_pdd_a['annual_pred_PDD'][stake]) / (1000)

    legend_text = "\n".join((
        r"$\mathrm{MAE}_{miniML}=%.3f, \mathrm{\rho}_{miniML}=%.2f, \mathrm{std}_{miniML}=%.2f$"
        % (
            mae_xgb,
            pearson_xgb,
            std_ml,
        ),
        (r"$\mathrm{MAE}_{PDD}=%.3f,\mathrm{\rho}_{PDD}=%.2f, \mathrm{std}_{PDD}=%.2f$"
         % (mae_pdd, pearson_pdd, std_pdd)),
    ))
    ax2.text(0.02,
             0.02,
             legend_text,
             transform=ax2.transAxes,
             verticalalignment="bottom",
             fontsize=16)

    ax2.tick_params(axis='both', labelsize=20)
    ax1.tick_params(axis='both', labelsize=20)
    ax2.set_title(stake)


def plotInputExtrm(best_combi, weights_t2m, weights_tp, stake_old, input_type,
                   vars_, ax1, ax2, ax3, ax4, extrm_years_t2m, extrm_years_tp):
    inputDF, target, xr_temppr = createSingleInput(best_combi,
                                                   weights_t2m,
                                                   weights_tp,
                                                   stake_old,
                                                   input_type,
                                                   vars_,
                                                   log=False,
                                                   unseen=True)

    inputDF_obs = inputDF.loc[target.index]
    inputDF['type obs'] = np.where(inputDF.index.isin(target.index), 'obs.',
                                   'unobs.')
    upper_t2m = np.percentile(inputDF_obs.t2m_mean.values, 99) + 0.5 * abs(
        np.std(inputDF_obs['t2m_mean'].values))
    lower_t2m = np.percentile(inputDF_obs.t2m_mean.values, 1) - 0.5 * abs(
        np.std(inputDF_obs['t2m_mean'].values))
    upper_tp = np.percentile(inputDF_obs.tp_tot.values, 99) + 0.5 * abs(
        np.std(inputDF_obs['tp_tot'].values))
    lower_tp = np.percentile(inputDF_obs.tp_tot.values, 1) - 0.5 * abs(
        np.std(inputDF_obs['tp_tot'].values))
    sns.scatterplot(inputDF,
                    x=inputDF.index,
                    y='t2m_mean',
                    hue='type obs',
                    palette=sns.color_palette(['#a3d1ff', 'grey']),
                    s=60,
                    ax=ax2)
    ax2.scatter(x=extrm_years_t2m,
                y=inputDF.loc[extrm_years_t2m].t2m_mean,
                label='unobs. extr',
                color='#FF0000',
                s=20)
    sns.lineplot(inputDF, x=inputDF.index, y='t2m_mean', color='grey', ax=ax2)

    ax2.axhline([upper_t2m], color='#FF0000', linestyle='dashed', linewidth=1)
    ax2.axhline([lower_t2m], color='#FF0000', linestyle='dashed', linewidth=1)
    ax2.legend(fontsize=15)
    ax2.set_title('T[May-Aug.]')

    sns.histplot(inputDF.t2m_mean.values,
                 label='all years',
                 alpha=0.8,
                 kde=True,
                 ax=ax1)
    sns.histplot(inputDF.loc[target.index].t2m_mean.values,
                 label='obs. years',
                 color='grey',
                 alpha=0.8,
                 kde=True,
                 ax=ax1)
    ax1.set_xlabel('t2m')
    ax1.axvline([upper_t2m], color='#FF0000', linestyle='dashed', linewidth=1)
    ax1.axvline([lower_t2m], color='#FF0000', linestyle='dashed', linewidth=1)
    ax1.legend(fontsize=15)
    ax1.set_title('T[May-Aug.]')

    # TP
    sns.scatterplot(inputDF,
                    x=inputDF.index,
                    y='tp_tot',
                    hue='type obs',
                    palette=sns.color_palette(['#a3d1ff', 'grey']),
                    s=60,
                    ax=ax4)
    sns.lineplot(inputDF, x=inputDF.index, y='tp_tot', color='grey', ax=ax4)
    ax4.scatter(x=extrm_years_tp,
                y=inputDF.loc[extrm_years_tp].tp_tot,
                label='unobs. extr',
                color='#FF0000',
                s=20)

    ax4.axhline([upper_tp], color='#FF0000', linestyle='dashed', linewidth=1)
    ax4.axhline([lower_tp], color='#FF0000', linestyle='dashed', linewidth=1)
    ax4.legend(fontsize=15)
    ax4.set_title('P[Oct.-Feb.]')

    sns.histplot(inputDF.tp_tot.values,
                 label='all years',
                 alpha=0.8,
                 kde=True,
                 ax=ax3)
    sns.histplot(inputDF.loc[target.index].tp_tot.values,
                 label='obs. years',
                 color='grey',
                 alpha=0.8,
                 kde=True,
                 ax=ax3)
    ax3.set_xlabel('tp')
    # add vertical line
    ax3.axvline([upper_tp], color='#FF0000', linestyle='dashed', linewidth=1)
    ax3.axvline([lower_tp], color='#FF0000', linestyle='dashed', linewidth=1)
    ax3.legend(fontsize=15)
    ax3.set_title('P[Oct.-Feb.]')
