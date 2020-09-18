# *********************************************************************************
# Lore, Copyright (c) 2020, Alliance for Sustainable Energy, LLC.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
#
# Redistributions of source code must retain the above copyright notice, this list
# of conditions and the following disclaimer.
#
# Redistributions in binary form must reproduce the above copyright notice, this
# list of conditions and the following disclaimer in the documentation and/or other
# materials provided with the distribution.
#
# Neither the name of the copyright holder nor the names of its contributors may be
# used to endorse or promote products derived from this software without specific
# prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.
# *********************************************************************************

big_number = 1.0e10
epsilon = 1.0e-10

# Alphabetize!
# TODO discuss whether battery, PV, persistence parameters warrant separate sections
schemas = {
    'power_plant': {},
    'weather': {},
    'pysam_daotk': {
        'defocus': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': 1,
                },
            'required': True,
            'meta': {'label': 'Field optical focus fraction [-]'}
            },
        'disp_pceff_expected': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': 1,
                },
            'required': True,
            'meta': {'label': 'Dispatch expected power cycle efficiency adj. [-]'}
            },
        'disp_tes_expected': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Dispatch expected TES charge level [MWht]'}
            },
        'disp_wpb_expected': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Dispatch expected power generation [MWe]'}
            },
        'e_ch_tes': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'TES charge state [MWht]'}
            },
        'helio_positions': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Heliostat position table [matrix]'}
            },
        'P_cycle': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'PC electrical power output, gross [MWe]'}
            },
        'P_out_net': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': -big_number,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Total electric power to grid [MWe]'}
            },
        'P_rec_heattrace': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Receiver heat trace parasitic load [MWe]'}
            },
        'q_dot_rec_inc': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Receiver incident thermal power [MWt]'}
            },
        'q_pc_startup': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'PC startup thermal energy [MWht]'}
            },
        'q_sf_inc': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Field incident thermal power [MWt]'}
            },
        'q_startup': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Receiver startup thermal energy consumed [MWt]'}
            },
        'Q_thermal': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': -big_number,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Receiver thermal power to HTF less piping loss [MWt]'}
            },
        'sf_adjust_out': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0,
                'max': 1,
                },
            'required': True,
            'meta': {'label': 'Field availability adjustment factor [-]'}
            },
        'T_pc_in': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': -50,
                'max': 1000,
                },
            'required': True,
            'meta': {'label': 'PC HTF inlet temperature [C]'}
            }
        }, # pysam_daotk
    'dispatch_opt': {
        'A_V': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery linear voltage model slope coeffifient'}
            }, 
        'alpha': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Conversion factor between unitless and monetary values [$]'}
            }, 
        'alpha_n': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'i-directional converter slope-intercept parameter'}
            }, 
        'alpha_p': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Bi-directional converter slope-intercept parameter'}
            }, 
        'alpha_pv': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': ''}
            }, 
        'B_V': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery linear voltage model intercept coefficient'}
            }, 
        'beta_n': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Bi-directional converter slope parameter'}
            }, 
        'beta_p': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Bi-directional converter slope parameter'}
            }, 
        'beta_pv': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': ''}
            }, 
        'C_B': { 
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery manufacturer-specified capacity [kAh]'}
            }, 
        'C_delta_w': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Penalty for change in power cycle  production [\$/\Delta-kWe]'}
            }, 
        'C_v_w': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Penalty for change in power cycle  production beyond designed limits [\$/\Delta-kWe]'}
            }, 
        'Cbc': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Operating cost of charging battery [$/kWhe]'}
            }, 
        'Cbd': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Operating cost of discharging battery [$/kWhe]'}
            }, 
        'Cbl': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Lifecycle cost for battery [$/lifecycle]'}
            }, 
        'Cchsp': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Penalty for power cycle hot start-up [$/start]'}
            }, 
        'Ccsb': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Operating cost of power cycle standby operation [$/kWht]'}
            }, 
        'Ccsu': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Penalty for power cycle cold start-up [$/start]'}
            }, 
        'Cpc': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Operating cost of power cycle [$/kWhe]'}
            }, 
        'Cpv': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Operating cost of photovoltaic field [$/kWhe]'}
            }, 
        'Crec': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Operating cost of heliostat field and receiver [$/kWht]'}
            }, 
        'Crhsp': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Penalty for receiver hot start-up [$/start]'}
            }, 
        'Crsu': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Penalty for receiver cold start-up [$/start]'}
            }, 
        'Delta': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': 24.0,
                },
            'required': True,
            'meta': {'label': 'length of each period (h)'}
            },
        'Delta_e': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': 8760.0,
                },
            'required': True,
            'meta': {'label': 'cumulative time elapsed at end of each period (h)'}
            }, 
        'delta_rs': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                },
            'required': True,
            'meta': {'label': 'Estimated fraction of period required for receiver start-up'}
            }, 
        'deltal': {
            'type': 'float',
            'min': 0.0,
            'max': 24.0,
            'required': True,
            'meta': {'label': 'Minimum time to start the receiver [hr]'}
            },
        'Ec': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Required energy expended to start cycle [kWh\sst]'}
            }, 
        'Ehs': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Heliostat field startup or shut down parasitic loss [kWh\sse]'}
            }, 
        'Er': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Required energy expended to start receiver [kWh\sst]'}
            }, 
        'eta_des': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Cycle nominal efficiency [-]'}
            }, 
        'etaamb': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                },
            'required': True,
            'meta': {'label': 'Cycle efficiency ambient temperature adjustment factor in period'}
            }, 
        'etac': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                },
            'required': True,
            'meta': {'label': 'Normalized condenser parasitic loss in period'}
            },
        'etap': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Slope of linear approximation of power cycle performance curve [kW\sse/kW\sst]'}
            }, 
        'Eu': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Thermal energy storage capacity [kWh\sst]'}
            }, 
        'I_avg': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Assumed average current for linearization of battery model [A]'}
            }, 
        'I_lower_n': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery min discharge current [A]'}
            }, 
        'I_upper_n': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery max discharge current'}
            }, 
        'I_upper_p': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery max charge current'}
            }, 
        'Lc': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Cycle heat transfer fluid pumping power per unit energy expended [kW\sse/kW\sst]'}
            },
        'Lr': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Receiver pumping power per unit power produced [kW\sse/kW\sst]'}
            },
        'N_csp': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': ''}
            }, 
        'num_periods': {
            'type': 'integer',
            'min': 1,
            'max': 8760,
            'required': True,
            'meta': {'label': 'number of time periods'}
            },
        'P': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Electricity sales price in period [$/kWhe]'}
            },
        'P_B_lower': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery min power rating [kWe]'}
            }, 
        'P_B_upper': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery max power rating [kWe]'}
            }, 
        'Qb': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Cycle standby thermal power consumption per period [kW\sst]'}
            }, 
        'Qc': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Allowable power per period for cycle start-up in period [kWt]'}
            },
        'Qin': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Available thermal power generated by the CSP heliostat field in period [kWt]'}
            },
        'Ql': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Minimum operational thermal power input to cycle [kW\sst]'}
            }, 
        'Qrl': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Minimum operational thermal power delivered by receiver [kWh\sst]'}
            }, 
        'Qrsb': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Required thermal power for receiver standby [kWh\sst]'}
            }, 
        'Qrsd': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Required thermal power for receiver shut down [kWht]'}
            }, 
        'Qru': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Allowable power per period for receiver start-up [kWht]'}
            },
        'Qu': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Cycle thermal power capacity [kW\sst]'}
            },
        'R_int': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Internal resistance of battery [Ohm]'}
            },
        'S_B_lower': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery min state of charge'}
            }, 
        'S_B_upper': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery max state of charge'}
            }, 
        's0': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'default': 0.0,
            'meta': {'label': 'Initial TES reserve quantity  [kWh\sst]'}
            }, 
        'soc0': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Battery initial state of charge [kWhe]'}
            },
        'ucsu0': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'default': 0.0,
            'meta': {'label': 'Initial cycle start-up energy inventory  [kWh\sst]'}
            }, 
        'ursu0': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'default': 0.0,
            'meta': {'label': 'Initial receiver start-up energy inventory [kWh\sst]'}
            },
        'W_delta_minus': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Power cycle ramp-down designed limit [kW\sse/h]'}
            }, 
        'W_delta_plus': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Power cycle ramp-up designed limit [kW\sse/h]'}
            }, 
        'W_u_minus': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Maximum power production in period t when stopping generation in period t+1  [kWe]'}
            }, 
        'W_u_plus': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Maximum power production when starting generation in period [kWe]'}
            }, 
        'W_v_minus': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Power cycle ramp-down violation limit [kW\sse/h]'}
            }, 
        'W_v_plus': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Power cycle ramp-up violation limit [kW\sse/h]'}
            }, 
        'Wb': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Power cycle standby operation parasitic load [kW\sse]'}
            }, 
        'wdot_s_pen': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'default': 0.0,
            'meta': {'label': 'penalty for difference between electrical power output vs. previous instances of the optimization model [$/kWhe]'}
            }, 
        'wdot_s_prev': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'default': 0.0,
            'meta': {'label': 'electrical power output from previous instances of the optimization model [kWhe]'}
            },
        'wdot0': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'default': 0.0,
            'meta': {'label': 'Initial power cycle electricity generation [kW\sse]'}
            },
        'Wdotl': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Minimum cycle electric power output [kW\sse]'}
            }, 
        'Wdotnet': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': big_number,
                },
            'required': True,
            'meta': {'label': 'Net grid transmission upper limit in period [kWe]'}
            }, 
        'Wdotu': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Cycle electric power rated capacity [kW\sse]'}
            }, 
        'Wh': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Heliostat field tracking parasitic loss [kWe]'}
            }, 
        'Wht': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Tower piping heat trace parasitic loss [kWe]'}
            }, 
        'Winv_lim': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Inverter max power (DC)'}
            }, 
        'Winvnt': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': ''}
            }, 
        'Wmax': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': False,
            'meta': {'label': 'Constant Max power to grid'}
            }, 
        'wpv_dc': {
            'type': 'list',
            'schema': {
                'type': 'float',
                'min': 0.0,
                'max': big_number,
                },
            'required': False,
            'meta': {'label': 'maximum DC power production from PV system in period [kWe]'}
            }, 
        'y0': {
            'type': 'boolean',
            'default': False,
            'meta': {'label': '1 if cycle is generating electric power initially, 0 otherwise'}
            }, 
        'ycsb0': {
            'type': 'boolean',
            'default': False,
            'meta': {'label': '1 if cycle is in standby mode initially, 0 otherwise'}
            }, 
        'ycsu0': {
            'type': 'boolean',
            'default': False,
            'meta': {'label': '1 if cycle is in starting up initially, 0 otherwise'}
            }, 
        'Yd': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Minimum required power cycle downtime [h]'}
            }, 
        'Yd0': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'default': 8760.0,
            'meta': {'label': 'duration that cycle has not been generating power (i.e., shut down or in standby mode) [h]'}
            },
        'yr0': {
            'type': 'boolean',
            'default': False,
            'meta': {'label': '1 if receiver is generating ``usable'' thermal power initially, 0 otherwise'}
            }, 
        'yrsb0': {
            'type': 'boolean',
            'default': False,
            'meta': {'label': '1 if receiver is in standby mode initially, 0 otherwise'}
            }, 
        'yrsu0': {
            'type': 'boolean',
            'default': False,
            'meta': {'label': '1 if receiver is in starting up initially, 0 otherwise'}
            }, 
        'Yu': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'required': True,
            'meta': {'label': 'Minimum required power cycle uptime [h]'}
            }, 
        'Yu0': {
            'type': 'float',
            'min': 0.0,
            'max': big_number,
            'default': 0.0,
            'meta': {'label': 'duration that cycle has been generating electric power [h]'}
            }
        } # dispatch_opt
    } # schemas