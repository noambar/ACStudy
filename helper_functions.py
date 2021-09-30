import warnings
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
from mne.stats.multi_comp import fdr_correction
from numpy import asarray, compress, not_equal, sqrt
from pandas import Series, DataFrame, concat
from pandas.core.common import isnull
from scipy.stats import distributions
from scipy.stats.stats import find_repeats, rankdata, tiecorrect

eps = 1e-10


def mkdirifnotexists(in_pths, chdir=False):
    def _mksingledir(in_pth):
        os.makedirs(in_pth, exist_ok=True)
        return in_pth
    
    
def _change_index(data, df, from_idx, to_idx):
    c_name = data.columns.names
    data = data.reset_index().merge(df.reset_index()[[from_idx, to_idx]], on=from_idx).set_index(to_idx).copy()
    del data[from_idx]
    data = data.loc[data.index.notnull()]
    data.columns.names = c_name
    return data


def pearsonr_rmna(x, y):
    try:
        X = x.values.ravel()
        Y = y.values.ravel()
    except:
        X = x.ravel()
        Y = y.ravel()
    mask = ~np.isnan(X) & ~np.isnan(Y)
    # mask array is now true where ith rows of df and dg are NOT nan.
    X = X[mask]  # this returns a 1D array of length mask.sum()
    Y = Y[mask]
    return pearsonr(X, Y)


def remove_rare_elements(df, rare_def=0.05, null=True):
    """ Will keep only rows with at least rare_def percent non-null (or zero) values.
        Assumes rows are elements and columns are samples."""
    if null:
        rows2keep = df.notnull().sum(1) > float(df.shape[1] * rare_def)
    else:
        df_min = df.min().min()
        rows2keep = (df > df_min).sum(1) > float(df.shape[1] * rare_def)
    print ('removing %0.3f of elements.' % (1 - float(rows2keep.sum()) / len(rows2keep)))
    return df.loc[rows2keep].copy()


def directed_mannwhitneyu(x, y, use_continuity=True):
    """
    Copy of scipy.stats.mannwhitneyu which multiplies the static by the direction.
    """
    x = asarray(x)
    y = asarray(y)
    n1 = len(x)
    n2 = len(y)
    ranked = rankdata(np.concatenate((x, y)))
    rankx = ranked[0:n1]  # get the x-ranks
    u1 = n1 * n2 + (n1 * (n1 + 1)) / 2.0 - np.sum(rankx, axis=0)  # calc U for x
    u2 = n1 * n2 - u1  # remainder is U for y
    bigu = max(u1, u2)
    smallu = min(u1, u2)
    t = tiecorrect(ranked)
    if t == 0:
        raise ValueError('All numbers are identical in amannwhitneyu')
    sd = np.sqrt(t * n1 * n2 * (n1 + n2 + 1) / 12.0)

    if use_continuity:
        # normal approximation for prob calc with continuity correction
        z = abs((bigu - 0.5 - n1 * n2 / 2.0) / sd)
    else:
        z = abs((bigu - n1 * n2 / 2.0) / sd)  # normal approximation for prob calc
    return (eps if smallu == 0 else smallu) * (-1 if smallu == u1 else 1), distributions.norm.sf(z)  # (1.0 - zprob(z))


from matplotlib.colors import Normalize
from matplotlib import colors as mcolors

def _get_scale_colors(cmaps, data, zero_is_middle=True, base_n=300, boundries=None, return_cmap=False):
    if boundries is None:
        data_plus_min = data - min(0, data.min())
        data_plus_min /= data_plus_min.max()
        min_max_ratio = abs(data.min() / float(data.max()))
    else:
        data_plus_min = data + abs(boundries[0])
        data_plus_min /= (abs(boundries[0]) + boundries[1])
        min_max_ratio = abs(boundries[0] / float(boundries[1]))
    if len(cmaps) == 1:
        return [cmaps[0](i) for i in data_plus_min]

    colors1 = cmaps[0](np.linspace(0., 1, int(base_n*min_max_ratio)))
    colors2 = cmaps[1](np.linspace(0., 1, base_n))
    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
    if return_cmap:
        return mymap
    return [mymap(i) for i in data_plus_min]

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

sns.palplot(_get_scale_colors([plt.cm.Blues_r, plt.cm.Reds], np.sort(np.random.uniform(-0.5, 0.5, 100)), boundries=[-1, 0.5] ))

from matplotlib.colors import LinearSegmentedColormap
cm_acs = LinearSegmentedColormap.from_list('acs', ['white', acs_color], N=1000)
cm_healthy = LinearSegmentedColormap.from_list('healthy', [healthy_color, 'white'], N=1000)

def add_text_at_corner(myplt, text, where='top right', **kwargs):
    legal_pos = ['top right', 'top left', 'bottom right', 'bottom left']
    if where not in legal_pos:
        print ("where should be one of: " + ', '.join(legal_pos))
        return
    topbottom = where.split()[0]
    rightleft = where.split()[1]
    if str(type(myplt)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>" or str(
            type(myplt)) == "<class 'mpl_toolkits.axes_grid1.parasite_axes.AxesHostAxes'>":
        x = myplt.get_xlim()
        y = myplt.get_ylim()
    elif str(type(myplt)) == "<type 'module'>":
        x = myplt.xlim()
        y = myplt.ylim()
    else:

        raise
    newaxis = {'left': x[0] + (x[1] - x[0]) * 0.01, 'right': x[1] - (x[1] - x[0]) * 0.01,
               'top': y[1] - (y[1] - y[0]) * 0.01, 'bottom': y[0] + (y[1] - y[0]) * 0.01}
    myplt.text(newaxis[rightleft], newaxis[topbottom], text, horizontalalignment=rightleft, verticalalignment=topbottom,
               **kwargs)
    
    
def add_null_indicator_column(df, null=np.nan, fillna=None, prefix='indicator'):
    """
    adding indicator columns to dataframe based on some value
    :param df:
    :param null:
    :param prefix:
    :return:
    """
    new_df = df.copy()
    for col in df.columns:
        if null is np.nan:
            if df[col].isnull().sum() > 0:
                new_df[prefix + '_' + col] = df[col].isnull().astype(int).copy()
                if fillna is not None:
                    new_df[col].fillna(fillna, inplace=True)
        else:
            if (df[col] == null).sum() > 0:
                new_df[prefix + '_' + col] = (df[col] == null).astype(int).copy()
                if fillna is not None:
                    new_df[col].replace(null, fillna, inplace=True)
    return new_df

def r2_score_rmna(x, y):
    try:
        X = x.values.ravel()
        Y = y.values.ravel()
    except:
        X = x.ravel()
        Y = y.ravel()
    mask = ~np.isnan(X) & ~np.isnan(Y)
    # mask array is now true where ith rows of df and dg are NOT nan.
    X = X[mask]  # this returns a 1D array of length mask.sum()
    Y = Y[mask]
    return r2_score(X, Y)

def spearmanr_minimal(x, y, return_values=(np.nan, np.nan), nan_policy='omit'):
    try:
        X = x.values.ravel()
        Y = y.values.ravel()
    except:
        X = x.ravel()
        Y = y.ravel()
    if len(set(X)) < 3 or len(set(Y)) < 3:
        return return_values
    else:
        return spearmanr(X, Y, nan_policy=nan_policy)
