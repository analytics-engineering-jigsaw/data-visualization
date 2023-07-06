import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

def build_grouped_by(table_name, col, target, engine, order_by_col = False):
    if order_by_col:
        query = f"""select {col}, sum({target}) total_amount from {table_name} group by {col} order by {col} asc"""
    else:
        query = f"""select {col}, sum({target}) total_amount from {table_name} group by {col} order by total_amount desc"""
    grouped = pd.read_sql(query, engine)
    return grouped

def build_grouped_bys(table_name, cols, target, engine, order_by_col = False):
    grouped_bys = []
    for col in cols:
        grouped = build_grouped_by(table_name, col, target, engine, order_by_col = False)
        grouped_bys.append(grouped)
    return grouped_bys

def print_grouped_by(grouped):
    selected_group = grouped[grouped.iloc[:, 0].values != None]
    plt.figure(figsize=(14, 2))
    plt.scatter(selected_group.iloc[:10, 0], selected_group.iloc[:10, 1])
    plt.show()

def print_grouped_bys(cols, grouped_bys):
    for col, group in zip(cols, grouped_bys):
        print(col)
        print_grouped_by(group)

def group_and_print(table_name, cols, target, engine, order_by_col = False):
    grouped_bys = build_grouped_bys(table_name, cols, target, engine, order_by_col)
    print_grouped_bys(cols, grouped_bys)