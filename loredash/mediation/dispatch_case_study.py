from genericpath import isfile
import pandas as pd
import numpy as np
import sys, os
import rapidjson
from pathlib import Path
import shutil
import time
import sqlite3
import csv
import datetime
sys.path.insert(1, os.path.join(sys.path[0], '..'))
#from loredash.mediation.mediator import Mediator

#id refers to each specific case study, it should contain an identifier which highlights...
#interesting qualties about the case.

#Path dict should include paths to: mediatior params, plant design params, weather file path.
#will be used to instantiate the mediator class run lore on the params
#e.g.
#
#path_dict = {
#    mediator_params = 'config/mediator_params.json,
#    plant_design_params = 'config/mediator_params.json'
#    weather_file_path = 'config/tmyfile.csv
#             }
parent_dir = str(Path(__file__).parents[1])+'/'
modify_paths = {'mediator_params':'config/mediator_params.json','plant_design_params':'config/mediator_params.json','weatherfile':'config/weatherfile.csv'}
def db_check():
    conn = sqlite3.connect(os.path.join(parent_dir, 'db.sqlite3'))
    cur = conn.cursor()
    try:
        cur.execute('''CREATE TABLE ui_dashboarddatarto (id, timestamp, actual, optimal, scheduled, field_operation_generated, field_operation_available)''')
        cur.execute('''CREATE TABLE ui_forecastsmarketdata (id, timestamp, market_forecast, ci_plus, ci_minus)''')
        cur.execute('''CREATE TABLE mediation_weatherdata (id, timestamp, dni, dhi, ghi, dew_point, temperature, pressure, wind_direction, wind_speed)''')
        cur.execute('''CREATE TABLE mediation_techdata (id, timestamp, E_tes_charged, eta_tower_thermal, eta_field_optical, W_grid_no_derate, tou, W_grid_with_derate, Q_tower_incident, Q_field_incident, pricing_multiple, dni, Q_tower_absorbed, mdot_tower, mdot_cycle)''')
    except:
        print('these tables exist!')
    conn.commit()
    return
class DispatchCaseStudy:
    def __init__(self, id, path_dict):
        self.id = id
        self.paths = path_dict
        with open(parent_dir+self.paths['mediator_params']) as f:
            self.mediator_params = rapidjson.load(f)
    def validate(self):
        time.sleep(60.0)
        print("Validated :)")
    def save_case(self,variable):
        parent_dir = str(Path().parent.absolute())+'/case studies'
        self.variable = variable
        home_dir = parent_dir+'/'+self.variable
        if not os.path.isdir(home_dir):
            os.mkdir(home_dir)
        case_dir = 'case studies/'+self.variable+'/'+self.id+' case study summary'
        if not os.path.isdir(case_dir):
            os.makedirs(case_dir)
        tech_save_file = os.path.join(case_dir,self.id+'_case_study_tech_outputs.json')
        if os.path.exists(tech_save_file):
            tech_save_file = os.path.join(case_dir,self.id+'_case_study_day2_tech_outputs.json')
        if os.path.exists('result.json'):
            os.rename('result.json',tech_save_file)
        parent_dir = str(Path().parent.absolute())

        shutil.copyfile(parent_dir+self.paths['plant_design_params'],os.path.join(case_dir,self.id+'_plant_design.json'))
        print('case study data recorded!')
        return
    def clean_up(self):
        config_dir = str(Path().parent.absolute())+'/config/'
        defaults_dir = config_dir+'defaults/'
        for f in os.listdir(config_dir):
            if os.path.isfile(config_dir+f) and 'dev.env' not in f and 'forecastfile.json' not in f:
                os.remove(config_dir+f)
        for file in os.listdir(defaults_dir):
            shutil.copyfile(defaults_dir+file,config_dir+file)
        return
    pass



#TODO Manipulate the output db's into writing to a json file which is then graphed using a pandas
# data frame comparing desired results OR simply read the results by peering into the database

#TODO add a function that will display a graph of all stored databases or something...
# maybe add this to a different file

#ramping penalties, plant_design "c_delta_w" value
#case study: vary the ramping penalties as a percentage of gross plant revenue, modify 'c_delta_w'
#case study: vary weather file/location
#check tuesday availability