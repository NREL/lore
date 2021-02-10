import csv
import json
import os
import re
import sqlite3
import sys

usage = 'import_data.py Usage: import_data data_file_path\n'

# Check to ensure the correct number of arguments are provided

num_args = len(sys.argv)
if(num_args < 2):
    print('Too few arguments: No file path to the data was provided.')
    sys.exit(usage)
elif(num_args > 2):
    print('Too many arguments.')
    sys.exit(usage)

# Check to ensure that the argument is a file path

user_filepath = sys.argv[1]
if (not os.path.isdir(user_filepath)):
    print('A valid directory was not provided.')
    sys.exit(usage)
else:
    user_filepath = os.path.abspath(user_filepath)
    print('Using data file path: {:s}'.format(os.path.abspath(user_filepath)))

# Select database

db_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(db_dir)

# Make database connection
with sqlite3.connect(os.path.join(db_dir, 'db.sqlite3')) as db:
    # Change to users file directory
    os.chdir(user_filepath)
    # Get a cursor for the database.
    cur = db.cursor()
    # Load the plant configuration from plant_config.json.
    with open('plant_config.json', 'r') as io:
        data = json.load(io)
        l = data['location']
        cur.execute('''
            INSERT INTO 'mediation_plantconfig' (name, latitude, longitude, elevation, timezone) 
            VALUES(?,?,?,?,?)''', 
            (data['name'], l['latitude'], l['longitude'], l['elevation'], l['timezone'])
        )
        print('Completed writes to mediation_plantconfig.')
    # Add the `ui-xxx.csv` files to the database.
    csv_match = re.compile(r'ui-.*-min\.csv')
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
    db.commit()
