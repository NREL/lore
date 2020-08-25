import os
import re
import csv
import sqlite3
import sys

usage = 'import_data.py Usage: import_data data_file_path\n'
num_args = len(sys.argv)
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Check to ensure the correct number of arguments are provided
if(num_args != 2):
    if(num_args < 2):
        print('Too few arguments: No file path to the data was provided.')
        sys.exit(usage)
    elif(num_args > 2):
        print('Too many arguments.')
        sys.exit(usage)

user_filepath = sys.argv[1]

#Check to ensure that the argument is a file path
if (not os.path.isdir(user_filepath)):
    print('A valid directory was not provided.')
    sys.exit(usage)
else:
    print('Using data file path: {:s}'.format(os.path.abspath(user_filepath)))

# Regex for the files to cycle through
file_match = re.compile(r'ui-.*-min\.csv')

# Set up connection
con = sqlite3.connect(os.path.join(curr_dir, "../db.sqlite3")) # change to 'sqlite:///your_filename.db'
cur = con.cursor()

# Change to users file directory
os.chdir(user_filepath)

# Itterate over the csv fils in this folder
for filename in os.listdir():
    if file_match.match(filename):
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
            elif re.match(r'.*solar.*', filename):
                table = 'ui_forecastssolardata'
                print('Writing to ' + table + '...')
                data_to_db = [((i+1),row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12],row[13],row[14],row[15],row[16]) for i,row in enumerate(reader)]
                cur.executemany('insert into ' + table + ' (id, timestamp, clear_sky, nam, nam_plus, nam_minus, rap, rap_plus, rap_minus, hrrr, hrrr_plus, hrrr_minus, gfs, gfs_plus, gfs_minus, ndfd, ndfd_plus, ndfd_minus) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);', data_to_db)
                print('Completed writes to ' + table + '.')

con.commit()
con.close()


