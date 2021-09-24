from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from scipy.special import expit, logit

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np




def calc_propensity_score_sklearn(data_df, ps_cols, TX_col, model=LogisticRegression(penalty='none', solver='saga'),
                                  CV=False):
    """
    Calculate Propensity scores using a sklearn clasifier model

    Parameters
    ----------
    data_df: DataFrame containing both X and T columns
    ps_cols: list of str, the columns to use for the propensity model
    TX_col: str, name of the Treatment columns.
    model: sklearn classifier model. Default is LR with no regularization.

    Returns
    -------
    ps_model: the model trainde
    ps_scores: list of propensity scores
    """
    ps_model = model.fit(X=data_df[ps_cols], y=data_df[TX_col])
    if CV:
        pscores = pd.Series(
            data=cross_val_predict(model, data_df[ps_cols], data_df[TX_col], method='predict_proba', cv=3)[:, 1],
            index=data_df.index)
    else:
        pscores = pd.Series(data=ps_model.predict_proba(data_df[ps_cols])[:, 1],
                            index=data_df.index)

    return ps_model, pscores


def plot_pscores(data_df, TX_col, pscore_col, TX_values=[0, 1], TX_labels=['A=0', 'A=1'],
                 ax=None, hist=False, normed=False, bin_resolution=0.01, labelsize=20, tick_labelsize=15,
                 title=None, palette=None, **kwargs):
    """
    Plot Propensity scores

    Parameters
    ----------
    data_df: DataFrame containing p_score column (propensity score column) and TX column (treatment column)
    TX_col: str, name of the Treatment columns.
    pscore_col: str, name of the column containing the propensity score.
    TX_values: values of T.
    TX_labels: labels for T.
    """

    idx_A0 = data_df[TX_col] == TX_values[0]
    idx_A1 = data_df[TX_col] == TX_values[1]

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    pscores0, pscores1 = data_df.loc[idx_A0, pscore_col].values, data_df.loc[idx_A1, pscore_col].values
    N_0, N_1 = pscores0.shape[0], pscores1.shape[0]

    bins = np.arange(0, 1 + bin_resolution, bin_resolution)

    ax.hist([pscores0, pscores1],
            label=['%s N=(%0.1d)' % (TX_labels[0], N_0), '%s N=(%0.1d)' % (TX_labels[1], N_1)],
            bins=bins, normed=normed, **kwargs)

    ax.legend(loc='best', fontsize=tick_labelsize)
    ax.set_title(title, fontsize=labelsize)
    ax.set_xlabel('Propensity scores', size=labelsize)
    ax.set_xlim(0, 1)
    ax.tick_params(labelsize=tick_labelsize)
    return ax


def find_propensity_score_matches(data_df, TX_col, pscore_col, TX_values=[0, 1],
                                  caliper=0.25, distance='linear', n_matches=1, random_seed=0):
    """
    Find propensity score matches

    Parameters
    ----------
    data_df: DataFrame containing  X,T and p_score columns
    ps_cols: list of str, the columns to use for the propemsity model
    TX_col: str, name of the Treatment columns.
    pscore_col: str, name of the column containing the propensity score.
    TX_values: values of T.
    TX_labels: labels for T.

    Returns
    -------
    X_matched: the input data_df with the additional column: matched_control_idx,
               which contains the idx of the matched control for treated subjects. If no match was found, then it contains NaN
    """

    idx_A0 = data_df[TX_col] == TX_values[0]
    idx_A1 = data_df[TX_col] == TX_values[1]

    X_T1, X_T0 = data_df[idx_A1].copy(), data_df[idx_A0].copy()
    X_T1['matched_control_idx'], X_T0.loc['matched_control_idx'] = None, None
    N1, N0 = len(X_T1), len(X_T0)
    g1, g0 = X_T1.loc[:, pscore_col].copy(), X_T0.loc[:, pscore_col].copy()

    # get caliper_value (default is 0.25 logit stdevs
    caliper_value = caliper * logit(data_df[pscore_col]).std()

    # Randomly permute the smaller group to get order for matching
    np.random.seed(random_seed)
    g1_idx_order = np.random.permutation(g1.index)
    for g1_idx in g1_idx_order:

        if distance == 'linear':  # linear propensity score (on the logits)
            dist = np.abs(logit(g1[g1_idx]) - logit(g0))
        else:  # regular propensity score
            dist = np.abs(g1[g1_idx] - g0)

        for i in range(n_matches):

            if dist.min() <= caliper_value:
                g0_idx = dist.idxmin()
                #                 if i == 0:
                #                     X_T1.loc[g1_idx, 'matched_control_idx'] = list([g0_idx])
                #                 else:
                #                     X_T1.loc[g1_idx, 'matched_control_idx'].append(g0_idx)
                X_T1.loc[g1_idx, 'matched_control_idx'] = 'matched'
                X_T0.loc[g0_idx, 'matched_control_idx'] = g1_idx
                g0 = g0.drop(g0_idx)
                dist = dist.drop(g0_idx)
            else:
                break
    #                 X_T1.loc[g1_idx, 'matched_control_idx'].append(np.nan)
    #         if len(X_T1.loc[g1_idx, 'matched_control_idx']) == 0:
    #             X_T1.loc[g1_idx, 'matched_control_idx'] = np.nan

    X_matched = pd.concat([X_T1, X_T0])
    return X_matched


def continuous_standadized_difference(X0, X1):
    return (X1.mean() - X0.mean()) / np.sqrt((X1.std() + X0.std()) / 2.)


def dichotomous_standadized_difference(X0, X1):
    P0, P1 = X0.mean(), X1.mean()
    return (P1 - P0) / np.sqrt((P1 * (1 - P1) + P0 * (1 - P0)) / 2.)


def plot_SMD_before_after_matching(smd_df, labels=('matched samples', 'unmatched samples'), ax=None,
                                   legend_fontsize=15, tick_fontsize=15, colors=['red', 'black'], y_tick_rotation=0,
                                   y_ticks=None, label_fontsize=20, figsize=(6, 5), grid=False):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.plot(smd_df.iloc[:, 0], range(smd_df.shape[0]), 'o', color=colors[0], label=labels[0])
    ax.plot(smd_df.iloc[:, 1], range(smd_df.shape[0]), 'o', color=colors[1], label=labels[1])
    ax.legend(loc='best', fontsize=legend_fontsize)
    ax.axvline(x=0, ls='--', color='grey')
    ax.axvline(x=0.1, ls='--', color='lightgrey')
    ax.axvline(x=-0.1, ls='--', color='lightgrey')
    ax.set_yticks(range(smd_df.shape[0]))
    if y_ticks is not None:
        assert len(y_ticks) == smd_df.shape[0]
        ax.set_yticklabels(y_ticks, rotation=y_tick_rotation)
    else:
        ax.set_yticklabels(smd_df.index, rotation=y_tick_rotation)
    ax.tick_params(labelsize=tick_fontsize)
    ax.set_xlabel('Standardized differences', fontsize=label_fontsize)
    if grid:
        ax.grid()
    fig.tight_layout()
    return ax