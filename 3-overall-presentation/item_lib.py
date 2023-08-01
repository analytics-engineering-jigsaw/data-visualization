import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

def build_grouped_by(df, col, target, agg = 'sum', agg_name = None, pct = True):
    grouped_data = df.groupby(col)[target].agg([agg]).sort_values(agg, ascending = False)
    if pct:
        grouped_data = (grouped_data/df[target].sum())
        cum_pct = grouped_data.cumsum()/grouped_data.sum()
        grouped_data = grouped_data.assign(cum_pct = cum_pct)
    if agg_name:
        grouped_data = grouped_data.rename(columns={agg: agg_name})
    return grouped_data.round(3)

def build_grouped_bys(df, cols, target, agg = 'sum', agg_name = None, pct = True):
    totals_by_col = [build_grouped_by(df, col, target, agg = 'sum', agg_name = agg_name) for col in cols]
    totals = dict(zip(cols, totals_by_col))
    return totals

def print_grouped_by(grouped, title = "", axis = '', limit = 10, y_range = [0, 1], y_col = 0):
    selected_group = grouped[grouped.iloc[:, 0].values != None]
    fig = plt.figure(figsize=(14, 2))
    plt.scatter(selected_group.index[:limit], selected_group.iloc[:limit, y_col])
    plt.ylim(y_range)
    plt.title(title)
    plt.show()
    
def build_and_print(df, cols, target, agg = 'sum', agg_name = "", limit = 10, y_range = [0, 1], y_col = 0):
    totals_by_col = [build_grouped_by(df, col, target, agg = 'sum', agg_name = agg_name)
                 for col in cols]
    totals = dict(zip(cols, totals_by_col))    
    [print_grouped_by(df, title = f"{agg_name} by {col}", y_range = y_range, limit = limit, y_col = y_col) for col, df in totals.items()]
    return totals

def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    if isinstance(fldnames,str):
        fldnames = [fldnames]
    for fldname in fldnames:
        fld = df[fldname]
        fld_dtype = fld.dtype
        if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
            fld_dtype = np.datetime64

        if not np.issubdtype(fld_dtype, np.datetime64):
            df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True, errors=errors)
        targ_pre = re.sub('[Dd]ate$', '', fldname)
        attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear']
        if time: attr = attr + ['Hour', 'Minute', 'Second']
        for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
        df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
        if drop: df.drop(fldname, axis=1, inplace=True)