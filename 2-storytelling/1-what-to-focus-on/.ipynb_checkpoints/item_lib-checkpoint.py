import pandas as pd
import numpy as np
import re

def build_grouped_by(df, col, target, agg = 'sum', agg_name = None, pct = True):
    grouped_data = df.groupby(col)[target].agg([agg]).sort_values(agg, ascending = False)
    if pct:
        grouped_data = (grouped_data/df[target].sum())
    if agg_name:
        grouped_data = grouped_data.rename(columns={agg: agg_name})
    return grouped_data

import matplotlib.pyplot as plt
def print_grouped_by(grouped, title = "", axis = '', limit = 10, y_range = [0, 1]):
    selected_group = grouped[grouped.iloc[:, 0].values != None].round(3)
    fig = plt.figure(figsize=(14, 2))
    plt.scatter(selected_group.index[:limit], selected_group.iloc[:limit, 0])
    plt.ylim(y_range)
    plt.title(title)
    plt.show()
    
def build_and_print(df, cols, target, agg = 'sum', agg_name = "", y_range = [0, 1], limit = 15):
    totals_by_col = [build_grouped_by(df, col, target, agg = 'sum', agg_name = agg_name, limit = limit)
                 for col in cols]
    totals = dict(zip(cols, totals_by_col))
    [print_grouped_by(df, title = f"{agg_name} by {col}", y_range = y_range) for col, df in totals.items()]
    return totals

def add_datepart(df, fldnames, drop=True, time=False, errors="raise"):
    """add_datepart converts a column of df from a datetime64 to many columns containing
    the information from the date. This applies changes inplace."""
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
    # https://github.com/lewtun/dslectures/blob/master/dslectures/structured.py#L65         