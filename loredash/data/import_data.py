import os
import re
import csv
import sqlite3
import sys
# from PyInquirer import prompt

#DEBUG
import pdb

usage = 'import_data.py Usage: import_data data_file_path\n'

# Check to ensure the correct number of arguments are provided
num_args = len(sys.argv)
if(num_args != 2):
    if(num_args < 2):
        print('Too few arguments: No file path to the data was provided.')
        sys.exit(usage)
    elif(num_args > 2):
        print('Too many arguments.')
        sys.exit(usage)

#Check to ensure that the argument is a file path
user_filepath = sys.argv[1]
if (not os.path.isdir(user_filepath)):
    print('A valid directory was not provided.')
    sys.exit(usage)
else:
    user_filepath=os.path.abspath(user_filepath)
    print('Using data file path: {:s}'.format(os.path.abspath(user_filepath)))

## Set up connection

# Select database
db_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(db_dir)

# db_match = re.compile(r'.*\.sqlite3') # Regex for available databses
# available_databases = [filename for filename in os.listdir() if db_match.match(filename)]

# # Get database selection

# questions = [
#   { 'type':'list',
#     'name':'database',
#     'message':"Select which SQLite3 database to import to:",
#     'choices': available_databases,
#             }
# ]
# answers = prompt(questions)

# Make database connection
# conn = sqlite3.connect(os.path.join(db_dir, answers['database']))
conn = sqlite3.connect(os.path.join(db_dir, 'db.sqlite3'))
cur = conn.cursor()
try:
    #cur.execute('''CREATE TABLE ui_dashboarddatarto (id, timestamp, actual, optimal, scheduled, field_operation_generated, field_operation_available)''')
    #cur.execute('''CREATE TABLE ui_forecastsmarketdata (id, timestamp, market_forecast, ci_plus, ci_minus)''')
    #cur.execute('''CREATE TABLE mediation_weatherdata (id, timestamp, dni, dhi, ghi, dew_point, temperature, pressure, wind_direction, wind_speed)''')
    #cur.execute('''CREATE TABLE mediation_techdata (id, timestamp, E_tes_charged, eta_tower_thermal, eta_field_optical, W_grid_no_derate, tou, W_grid_with_derate, Q_tower_incident, Q_field_incident, pricing_multiple, dni, Q_tower_absorbed, mdot_tower, mdot_cycle, defocus, clearsky, op_mode_1, op_mode_2, op_mode_3, disp_solve_state, disp_objective, disp_qsf_expected, disp_qsfprod_expected, disp_wpb_expected)''')
    cur.execute('''CREATE TABLE mediation_SolarForecastData (forecast_made,forecast_for,clear_sky,ratio)''')

except:
    print('these tables exist!')
conn.commit() # Save the changes made to the database
conn.close()  # Close the databse connection
"""
# Change to users file directory
os.chdir(user_filepath)

# Regex for the files to cycle through
csv_match = re.compile(r'ui-.*-min\.csv')

## Iterate over the csv files in this folder
for filename in [f for f in os.listdir() if csv_match.match(f)]:
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)

        if re.match(r'.*dashboard.*', filename):
            table = 'ui_dashboarddatarto'
            print('Writing to ' + table + '...')
            data_to_db = [((i+1),row[0],row[1],row[2],row[3],row[4],row[5]) for i,row in enumerate(reader)]
            cur.executemany('insert into ' + table + ' (id, timestamp, actual, optimal, scheduled, field_operation_generated, field_operation_available) values (?,?,?,?,?,?,?);', data_to_db)
            print('Completed writes to ' + table + '.')

        elif re.match(r'.*market.*', filename):
            table = 'ui_forecastsmarketdata'
            print('Writing to ' + table + '...')
            data_to_db = [((i+1),row[0],row[1],row[2],row[3]) for i,row in enumerate(reader)]
            cur.executemany('insert into ' + table + ' (id, timestamp, market_forecast, ci_plus, ci_minus) values (?,?,?,?,?);', data_to_db)
            print('Completed writes to ' + table + '.')

conn.commit() # Save the changes made to the database
conn.close()  # Close the databse connection


"""