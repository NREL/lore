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


