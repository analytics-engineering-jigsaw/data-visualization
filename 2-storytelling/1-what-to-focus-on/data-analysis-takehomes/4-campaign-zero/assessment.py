import pandas as pd

df = pd.read_csv('data/police_killings.csv', low_memory=False)
df2 = pd.read_csv('data/race_eth_by_state_2020_census.csv', low_memory=False)
df2.rename(columns={'NAME': 'state', 'HISPANIC': 'Hispanic', 'WHITE': 'White', 'BLACK': 'Black', 'NATIVE_AMERICAN': 'Native_american',
     'ASIAN': 'Asian', 'PACIFIC_ISLANDER': 'Pacific_islander', 'OTHER': 'Other'},  inplace=True)
df2['state'] = df2['state'].apply(lambda x: x[:2].upper())
df['date'] = pd.DatetimeIndex(df['date']).year
merged = pd.merge(df, df2, on='state')
census_incidents = merged[['state', 'encounter_type', 'race', 'Black','White','Hispanic',
 'Native_american', 'Asian', 'Pacific_islander', 'Other', 'TOTAL']]

#read csv files
def read_csv(csv):
    df = pd.read_csv(csv, low_memory=False)
    return df

#How many traffic stop-involved police killings (TSPKs) are there in total? 
def traffic_stops():
    encounters = df['encounter_type']
    framed_encounters = encounters.to_frame()
    selected_encounters = framed_encounters[framed_encounters['encounter_type'] == 'Traffic Stop']
    total_TPSK = selected_encounters.value_counts()
    return total_TPSK 

    #595

# What proportion of all police killings do TSPKs comprise?
def traffic_stop_proportion():
    traffic_stops_number = traffic_stops()
    victims = df['name'] 
    total_killings = len(victims)
    result = total_killings / traffic_stops_number
    return int(result)

    #17%


#What is the total number of incidents each year? 
def total_number_incidents_per_year_breakdown():
    yearly_breakdown = df['date'].value_counts()
    return yearly_breakdown

    # 2021.0    1149
    # 2018.0    1145
    # 2020.0    1132
    # 2015.0    1103
    # 2019.0    1097
    # 2017.0    1094
    # 2013.0    1087
    # 2016.0    1070
    # 2014.0    1049
    # 2022.0     416


#total number of incidents per year w/ year input
def total_per_year_breakdown(year):
    year_data = df[['date', 'agency_responsible', 'name']]
    yearly_breakdown = year_data.set_index('date').filter(like=str(year), axis=0)
    year_count = yearly_breakdown.groupby(['date'], as_index=False)['name'].size()
    return year_count

#incidents per year with racial breakdown
def year_race_breakdown():
    data = df[['date', 'race', 'name']]
    selected_data = data.groupby(['date','race'])['name'].size()
    return selected_data

#     date    race            
# 2013.0  Asian                19
#         Black               291
#         Hispanic            169
#         Native American       5
#         Pacific Islander      2


#total number of incidents per race breakdown
def racial_breakdown():
    number_of_incidents = df['race'].value_counts()
    return number_of_incidents

# White               4430
# Black               2555
# Hispanic            1809
# Unknown race         996
# Asian                145
# Native American      139
# Pacific Islander      60


#total number of incidents per race w/ race input
def racial_breakdown_a(race:str):
    incidents = df['race']
    framed_incidents = incidents.to_frame()
    incident_race_list = framed_incidents[framed_incidents['race'] == race]
    number_of_incidents = len(incident_race_list)
    return number_of_incidents

#What are the top 3 agencies responsible for incidents
def top_agency_count():
    agency_data = df['agency_responsible'].value_counts()
    return agency_data[:3]

# Los Angeles Police Department              150
# Phoenix Police Department                  136
# Los Angeles County Sheriff's Department    122

#Agencies responsible for Traffic Stop Killings
def agency_encounter_type():
    agency_data = df[['agency_responsible', 'encounter_type']]
    selected_agency_data = agency_data[agency_data['encounter_type'] == 'Traffic Stop']
    return selected_agency_data.value_counts()

# California Highway Patrol                 Traffic Stop      9
# Los Angeles County Sheriff's Department   Traffic Stop      8
# Arizona Department of Public Safety       Traffic Stop      7
# Las Vegas Metropolitan Police Department  Traffic Stop      6
# Phoenix Police Department                 Traffic Stop      5

#output to csv
def output_to_csv(filename, data_name):
    name = data_name
    csv_output = filename.to_csv(f'data/{name}.csv')
    return csv_output

#totals of TSPK's per state per year broken down by race then output to .csv file
def totals_by_state():
    state_data = df[['date', 'agency_responsible', 'name', 'encounter_type', 'race', 'state']]
    filtered_data = state_data[state_data['encounter_type'] == 'Traffic Stop']
    selected_data = filtered_data.groupby(['state', 'race'])['name'].size()
    selected_data = output_to_csv(selected_data, 'tspk_abs')
    return selected_data


def total_per_capita_state():
    state_data = df[['date', 'agency_responsible', 'name', 'encounter_type', 'race', 'state']]
    filtered_data = state_data[state_data['encounter_type'] == 'Traffic Stop']
    breakpoint()
    pass
   
#remove spaces from a column
def strip_spaces(df,column_name):
    df[column_name] = df[column_name].str.strip()
    return df

#drop characters after comma
def remove_after_comma(df, column_name):
    df[column_name] = df[column_name].str.split(',').str[0]
    return df

#clean specified column (remove space / drop characters after comma)
def clean_column(df, column_name):
    strip_spaces(df, column_name)
    remove_after_comma(df, column_name)
    return df

#column cleaned agency TPSK function
def cleaned_agency_encounter_type():
    agency_data = clean_column(df, 'agency_responsible')[['agency_responsible', 'encounter_type']]
    selected_agency_data = agency_data[agency_data['encounter_type'] == 'Traffic Stop']
    return selected_agency_data.value_counts()

# Georgia State Patrol                      Traffic Stop      10
# California Highway Patrol                 Traffic Stop       9
# Los Angeles County Sheriff's Department   Traffic Stop       8
# Arizona Department of Public Safety       Traffic Stop       7
# Las Vegas Metropolitan Police Department  Traffic Stop       6










    