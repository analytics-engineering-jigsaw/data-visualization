import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine

user_cols = ['user_age', 'user_gender', 'user_id']
order_cols = ['order_single_item', 'order_date', 'order_id', 'order_item_count']
timing_cols = ['order_day_part', 'order_date']

marketing_cols = ['order_channel', 'order_referrer', 'order_ad_shown', 'order_is_freeshipping', 'order_coupon_code', 'user_loyalty_program']
location_cols = ['order_store_city', 'order_store_id']
product_cols = ['list_sku', 'list_sku_category', 'list_sku_flavor'] 


def format_data(df):
    df = df.assign(order_date = pd.to_datetime(df['order_date']))
    df.columns = [col.lower() for col in df.columns]
    return df

def insert_data(df, engine):
    df.to_sql('raw_orders', engine, if_exists = 'replace')
    return df

def load_data():
    engine = create_engine('postgresql://@localhost:5432/willy_wonka')
    df = pd.read_sql('select * from raw_lineitems', engine)
    return engine, df

engine, df = load_data()

def print_grouped_bys(cols, grouped_bys):
    for col, group in zip(cols, grouped_bys):
        print(col)
        print_grouped_by(group)


def print_grouped_by(grouped):
    selected_group = grouped[grouped.iloc[:, 0].values != None]
    plt.figure(figsize=(14, 2))
    plt.scatter(selected_group.iloc[:10, 0], selected_group.iloc[:10, 1])
    plt.show()

# def build_grouped_by(df, col):
#     grouped = df.groupby([col]).sum().sort_values('order_value', ascending = False)['order_value']
#     return grouped

def build_grouped_by(table_name, col, engine):
    query = f"""select {col}, sum(order_value) total_order_value from {table_name} group by {col} order by total_order_value desc"""
    grouped = pd.read_sql(query, engine)
    return grouped



def build_grouped_bys(table_name, cols, engine):
    grouped_bys = []
    for col in cols:
        grouped = build_grouped_by(table_name, col, engine)
        grouped_bys.append(grouped)
    return grouped_bys

def group_and_print(table_name, cols, engine):
    grouped_bys = build_grouped_bys(table_name, cols, engine)
    print_grouped_bys(cols, grouped_bys)



