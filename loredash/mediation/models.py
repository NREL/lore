from django.db import models
from django.conf import settings

class TechData(models.Model):                                                                               # SSC variable name [pre unit conversion]:
    timestamp = models.DateTimeField(verbose_name="Timestep end", primary_key=True)                         #   time_hr (end of timestep) [hr]
    E_tes_charged = models.FloatField(verbose_name="TES charge state [kWht]", default=None)                 #   e_ch_tes [MWht]
    eta_tower_thermal = models.FloatField(verbose_name="Tower thermal efficiency [-]", default=None)        #   eta_therm [-]
    eta_field_optical = models.FloatField(verbose_name="Field optical efficiency [-]", default=None)        #   eta_field [-]
    W_grid_no_derate = models.FloatField(verbose_name="Power to grid without derate [kWe]", default=None)   #   P_out_net [MWe]
    tou = models.FloatField(verbose_name="TOU value [-]", default=None)                                     #   tou_value [-]
    W_grid_with_derate = models.FloatField(verbose_name="Power to grid with derate [kWe]", default=None)    #   gen [kWe]
    Q_tower_incident = models.FloatField(verbose_name="Tower incident thermal power [kWt]", default=None)   #   q_dot_rec_inc [MWt]
    Q_field_incident = models.FloatField(verbose_name="Field incident thermal power [kWt]", default=None)   #   q_sf_inc [MWt]
    pricing_multiple = models.FloatField(verbose_name="Pricing multiple [-]", default=None)                 #   pricing_mult [-]
    dni = models.FloatField(verbose_name="DNI [W/m2]", default=None)                                        #   beam [W/m2]
    Q_tower_absorbed = models.FloatField(verbose_name="Tower HTF heat rate [kWt]", default=None)            #   Q_thermal [MWt]
    mdot_tower = models.FloatField(verbose_name="Tower HTF mass flow rate [kg/s]", default=None)            #   m_dot_rec [kg/s]
    mdot_cycle = models.FloatField(verbose_name="Cycle HTF mass flow rate [kg/s]", default=None)            #   m_dot_pc [kg/s]

    def __str__(self):
        """shown when entry is generically queried"""
        return str(self.timestamp)

class PlantStateData(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestep end", primary_key=True)                                   # time_hr (end of timestep) [hr]
    is_field_tracking = models.FloatField(verbose_name="Is field tracking? [-]", default=None)                        # is_field_tracking_init [-]
    receiver_mode = models.FloatField(verbose_name="Receiver operating mode [-]", default=None)                       # rec_op_mode_initial [-]
    dt_rec_startup_remain = models.FloatField(verbose_name="Receiver startup time remaining [hr]", default=None)      # rec_startup_time_remain_init [hr]
    dE_rec_startup_remain = models.FloatField(verbose_name="Receiver startup energy remaining [kWh]", default=None)   # rec_startup_energy_remain_init [Wh]
    # dt_rec_current_mode = models.FloatField(verbose_name="Time receiver's been in current state [hr]", default=None)  # disp_rec_persist0 [hr]
    # dt_rec_not_on = models.FloatField(verbose_name="Time receiver's not been on (off or startup) [hr]", default=None) # disp_rec_off0 [hr]
    sf_adjust = models.FloatField(verbose_name="Solar field percent unavailable (latest) [%]", default=None)          # sf_adjust:hourly [%]
    T_cold_tank = models.FloatField(verbose_name="Cold tank temperature [C]", default=None)                           # T_tank_cold_init [C]
    T_hot_tank = models.FloatField(verbose_name="Hot tank temperature [C]", default=None)                             # T_tank_hot_init [C]
    Frac_avail_hot_tank = models.FloatField(verbose_name="Fraction of available storage in hot tank [%]", default=None) # csp_pt_tes_init_hot_htf_percent [%]
    cycle_mode = models.FloatField(verbose_name="Initial cycle operating mode [-]", default=None)                     # pc_op_mode_initial [0=startup, 1=on, 2=standby, 3=off, 4=startup_controlled]
    dt_cycle_startup_remain = models.FloatField(verbose_name="Cycle startup time remaining [hr]", default=None)       # pc_startup_time_remain_init [hr]
    dE_cycle_startup_remain = models.FloatField(verbose_name="Cycle startup energy remaining [kWh]", default=None)    # pc_startup_energy_remain_initial [kWh]
    # dt_cycle_current_mode = models.FloatField(verbose_name="Time cycle's been in current state [hr]", default=None)   # disp_pc_persist0 [hr]
    # dt_cycle_not_on = models.FloatField(verbose_name="Time cycle's not been on (off, startup, or standby) [hr]", default=None) # disp_pc_off0 [hr]
    W_cycle = models.FloatField(verbose_name="Cycle electricity generation [kWe]", default=None)                      # wdot0 [MWe]
    Q_cycle = models.FloatField(verbose_name="Cycle thermal input [kWt]", default=None)                               # qdot0 [MWt]

    def __str__(self):
        """shown when entry is generically queried"""
        return str(self.timestamp)

class WeatherData(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestep end", primary_key=True)
    dni = models.FloatField(verbose_name="Direct normal irradiance [W/m2]", default=None)
    dhi = models.FloatField(verbose_name="Diffuse horizontal irradiance [W/m2]", default=None)
    ghi = models.FloatField(verbose_name="Global horizontal irradiance [W/m2]", default=None)
    dew_point = models.FloatField(verbose_name="Dew point [C]", default=None)
    temperature = models.FloatField(verbose_name="Ambient dry bulb temperature [C]", default=None)
    pressure = models.FloatField(verbose_name="Ambient atmospheric pressure [mbar]", default=None)
    wind_direction = models.FloatField(verbose_name="Horizontal wind direction [deg]", default=None)
    wind_speed = models.FloatField(verbose_name="Horizontal wind speed [m/s]", default=None)

    def __str__(self):
        """shown when entry is generically queried"""
        return str(self.timestamp)

class SolarForecastData(models.Model):
    timestamp = models.DateTimeField(verbose_name="Timestep end", primary_key=True)
    dni = models.FloatField(verbose_name="Direct normal irradiance [W/m2]", default=None)
    dhi = models.FloatField(verbose_name="Diffuse horizontal irradiance [W/m2]", default=None)
    ghi = models.FloatField(verbose_name="Global horizontal irradiance [W/m2]", default=None)
    # dew_point = models.FloatField(verbose_name="Dew point [C]", default=None)
    temperature = models.FloatField(verbose_name="Ambient dry bulb temperature [C]", default=None)
    # pressure = models.FloatField(verbose_name="Ambient atmospheric pressure [mbar]", default=None)
    # wind_direction = models.FloatField(verbose_name="Horizontal wind direction [deg]", default=None)
    wind_speed = models.FloatField(verbose_name="Horizontal wind speed [m/s]", default=None)
    clear_sky = models.FloatField(verbose_name="Clear Sky [W/m2]", default=None)
    ratio = models.FloatField(verbose_name="Ratio [Predicted/Clear Sky]", default=None)
    dni_10 = models.FloatField(verbose_name="Clear Sky 10% probability [W/m2]", default=None)
    dni_25 = models.FloatField(verbose_name="Clear Sky 25% probability [W/m2]", default=None)
    dni_50 = models.FloatField(verbose_name="Clear Sky 50% probability [W/m2]", default=None)
    dni_75 = models.FloatField(verbose_name="Clear Sky 75% probability [W/m2]", default=None)
    dni_90 = models.FloatField(verbose_name="Clear Sky 90% probability [W/m2]", default=None)
    def __str__(self):
        return str(self.timestep)
