import os

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