from django.conf import settings
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from voluptuous import Schema, Required, Optional, Range, And, Or, DefaultTo, SetTo, Any, Coerce, Maybe, ALLOW_EXTRA, All, REMOVE_EXTRA, Number, Invalid


kBigNumber = 1.0e10
kEpsilon = 1.0e-10


def validate(values, schema, **kwargs):
    """
    Checks a dictionary of values against a validation schema

    values      one-level dictionary of values
    schema      one-level dictionary of validation criteria with keys corresponding
                 to variable names
    **kwargs    other named arguments for the validate() method

    The validation returns:
        - All of the values given that pass validation
        - Default values (e.g., None) for values not passing validation (including values originally equal to None)
        - Different default values (e.g., None) for missing values if they are required
        - Provided values not in the schema dictionary (they are not removed nor cause an exception) -> see 'extra' and 'required'
    The validation fails if:
        - There is a missing required value and no default is specified (see default=)
        - A value fails validation and no respective default is specified (see SetTo=)

    Background:
        - Cerebus has the best and most comprehensive features and syntax, but is unbelievably slow.
          It doesn't appear speed is an issue with any other validators, see: https://validx.readthedocs.io/en/latest/benchmarks.html
          However, it makes me hesistant to create my own validator.
        - Some other validators have similarly good syntax, but lack the needed features.
        - Voluptuous has all the needed features, is relatively fast, second biggest community, very flexible (even more so than Cerebus) but
          has a more difficult to read programming syntax compared to a markup-like one

    Notes:
        - See documentation: https://alecthomas.github.io/voluptuous/docs/_build/html/voluptuous.html
        - Designated invalid values for the candidate databases:
            PostgreSQL = NaN or NULL    # NaN is a special numerical value, NULL is a missing value
            SQLite3 = NULL              # Use NULL, as NaN are converted to NULL anyway
                                        #  (NULLs may be returned as 0's, so always type checking is recommended)
        - Python does not have NULL, rather it uses the None object.
        - Python NaN is created by float("NaN") and can be tested for using math.isnan()
    """

    try:
        update = settings.DEBUG    # don't let missing required variables cause a failure while debugging
    except:
        update = True       # set to true if not calling from Django framework

    validated_values = schema(values)
    sorted(validated_values)      # sort keys alphabetically
    return validated_values


# Normalization (coercion) functions
def MwToKw(mw):
    try:
        mw = float(mw)
    except:
        return None
    else:
        return mw * 1.e3



# **The following is the schema format for the Voluptuous validation package we are now using
#   Look to:
#    - verify within a min and max
#    - convert values (e.g., MW to kW)
#    - set a default value if key is missing (i.e., None)
#    - change invalid values to another (i.e., None)

#   Random Notes:
#    - Clamp() sets an out-of-bounds value to the nearest limit

ssc_schema = Schema( All(
        # First pass for renaming and normalization
        # Explanation:
        # - If multiple variables are present that are renamed to the same key, the latter entry in the input data dictionary overwrites the earlier one(s)
        # - A key value of 'object' results in the value being accepted as-is
        # Examples:
        # {
        # Optional(And('original_name_variation_MW', SetTo('new_name'))): [Coerce(MwToKw)],
        # Optional(And('original_name_kW', SetTo('new_name'))): object,
        # },
        {
            ## DO I WANT TO CONVERT THE MW TO kW FOR ALL THE VARIABLES??
        },

        # Second pass for validation
        # Explanation:
        # 1. "Required('key_name', default=[None])"  ->  Require 'key_name' be present. If not present, set to the one-entry list: [None]
        # 2. Specify the key's value is a list by enclosing conditions in brackets
        # 3. ": [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))]"  ->
        #    For each list value, try to coerce to a float and then verify it is within a min/max range. If either throws exception, set value to None.
        # 4. ": [Any(And(Coerce(MwToKw), float, Range(min=0, max=kBigNumber)), SetTo(None))]"  ->
        #    You need the 'float' because Range() can't be passed a None value. 'Coerce(float)' from above, and 'float' here both throw an
        #    exception if given a None, which is what you want in order to move on to the second Any() parameter, which is 'SetTo(None)'
        # Examples:
        # {
        # Required('missing_name', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        # Required('none_value', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        # Required('new_name', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        # Required('unique_name_MW', default=[None]): [Any(And(Coerce(MwToKw), float, Range(min=0, max=kBigNumber)), SetTo(None))],
        # Required('lots_of_values_MW', default=[None]): [Any(And(Coerce(MwToKw), float, Range(min=0, max=kBigNumber)), SetTo(None))],
        # },
        
        # Alphabetize!
        # The current names, units and datatype are those directly from ssc
        {
        Required('beam', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1600.)), SetTo(None))],                     # DNI [W/m2]
        Required('defocus', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1.)), SetTo(None))],                     # Field optical focus fraction [-]
        Required('disp_pceff_expected', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1.)), SetTo(None))],         # Dispatch expected power cycle efficiency adj. [-]
        Required('disp_tes_expected', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],   # Dispatch expected TES charge level [MWht]
        Required('disp_wpb_expected', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],   # Dispatch expected power generation [MWe]
        Required('e_ch_tes', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],            # TES charge state [MWht]
        Required('eta_field', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1.)), SetTo(None))],                   # Field optical efficiency [-]
        Required('eta_therm', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1.)), SetTo(None))],                   # Tower thermal efficiency [-]
        Required('gen', default=[None]): [Any(And(Coerce(float), Range(min=-kBigNumber, max=kBigNumber)), SetTo(None))],        # Power to grid with derate [kWe]
        Required('helio_positions', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],     # Heliostat position table [matrix]
        Required('hot_tank_htf_percent_final', default=[None]): [Any(And(Coerce(float), Range(min=0, max=100)), SetTo(None))],  # Final percent of maximum hot tank mass [%]
        Required('is_field_tracking_final', default=[None]): [Any(And(Coerce(float), Range(min=0, max=1)), SetTo(None))],       # Heliostat field tracking at end of timestep? (1 = true) [-]
        Required('P_cycle', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],             # PC electrical power output, gross [MWe]
        Required('P_out_net', default=[None]): [Any(And(Coerce(float), Range(min=-kBigNumber, max=kBigNumber)), SetTo(None))],  # Total electric power to grid [MWe]
        Required('P_rec_heattrace', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],     # Receiver heat trace parasitic load [MWe]
        Required('pc_op_mode_final', default=[None]): [Any(And(Coerce(float), Range(min=0, max=4)), SetTo(None))],              # Final cycle operation mode 0:startup, 1:on, 2:standby, 3:off, 4:startup_controlled [-]
        Required('pc_startup_energy_remain_final', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],   # Final cycle startup energy remaining [kWh]
        Required('pc_startup_time_remain_final', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],     # Final cycle startup time remaining [hr]
        Required('pricing_mult', default=[None]): [Any(And(Coerce(float), Range(min=-kBigNumber, max=kBigNumber)), SetTo(None))], # Pricing multiple [-]
        Required('q_dot_rec_inc', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],       # Receiver incident thermal power [MWt]
        Required('q_pc_startup', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],        # PC startup thermal energy [MWht]
        Required('q_sf_inc', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],            # Field incident thermal power [MWt]
        Required('q_startup', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],           # Receiver startup thermal energy consumed [MWt]
        Required('Q_thermal', default=[None]): [Any(And(Coerce(float), Range(min=-kBigNumber, max=kBigNumber)), SetTo(None))],  # Receiver thermal power to HTF less piping loss [MWt]
        Required('rec_op_mode_final', default=[None]): [Any(And(Coerce(float), Range(min=0, max=2)), SetTo(None))],             # Final receiver operating mode 0: off, 1: startup, 2: on [-]
        Required('rec_startup_energy_remain_final', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],  # Final receiver startup energy remaining [W-hr]
        Required('rec_startup_time_remain_final', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],    # Final receiver startup time remaining [hr]
        Required('sf_adjust_out', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1.)), SetTo(None))],               # Field availability adjustment factor [-]
        Required('T_pc_in', default=[None]): [Any(And(Coerce(float), Range(min=-50., max=1000.)), SetTo(None))],                # PC HTF inlet temperature [C]
        Required('T_tes_cold', default=[None]): [Any(And(Coerce(float), Range(min=-273.15, max=800)), SetTo(None))],            # TES cold temperature at end of timestep [C]
        Required('T_tes_hot', default=[None]): [Any(And(Coerce(float), Range(min=-273.15, max=800)), SetTo(None))],             # TES hot temperature at end of timestep [C]
        Required('time_hr', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],             # Timestep end [hr]
        Required('tou_value', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],           # Time-of-use value [-]
        },
    ),
        extra=ALLOW_EXTRA,          # ALLOW_EXTRA = keys in data that are undefined in schema won't cause exception--overrides default behavior of triggering an exception
                                    # REMOVE_EXTRA = keys in data that are undefined in schema will be removed--no exception will be thrown
        required=False              # False = data is not required for all defined schema keys, but this can be overridden per schema key using Required()
                                    # THIS DOESN'T SEEM TO WORK: True = data is required for all defined schema keys unless overriden using Optional()
)


def list_lengths_must_match(data_dict):
    list_lengths = [len(data_dict[key]) for key in data_dict if isinstance(data_dict[key], list)]
    if len(set(list_lengths)) > 1:
        raise Invalid('list lengths must match')
    return data_dict

weather_schema = Schema( All(
        # An exception will be thrown if any value is missing or invalid
        # Number(scale=0) throws exception if number has a non-zero decimal
        {
        Required('tz'): And(Number(scale=0), Coerce(int), Range(min=-12, max=12)),        # timezone [hr]
        Required('elev'): And(Coerce(float), Range(min=-500, max=9000)),                  # elevation [m]
        Required('lat'): And(Coerce(float), Range(min=-90, max=90)),                      # latitude [deg]
        Required('lon'): And(Coerce(float), Range(min=-180, max=180)),                    # longitude [deg]
        Required('year'): [And(Number(scale=0), Coerce(int), Range(min=0, max=3000))],    # year [-]
        Required('month'): [And(Number(scale=0), Coerce(int), Range(min=0, max=12))],     # month [-]
        Required('day'): [And(Number(scale=0), Coerce(int), Range(min=0, max=31))],       # day [-]
        Required('hour'): [And(Number(scale=0), Coerce(int), Range(min=0, max=24))],      # hour [hr]
        Required('minute'): [And(Number(scale=0), Coerce(int), Range(min=0, max=60))],    # minute [minute]
        Required('dn'): [And(Coerce(float), Range(min=0, max=1800))],                     # timezone [W/m2]
        Required('df'): [And(Coerce(float), Range(min=0, max=1000))],                     # timezone [W/m2]
        Required('gh'): [And(Coerce(float), Range(min=0, max=1800))],                     # timezone [W/m2]
        Required('wspd'): [And(Coerce(float), Range(min=0, max=110))],                    # timezone [m/s]
        Required('tdry'): [And(Coerce(float), Range(min=-100, max=150))],                 # timezone [C]
        },
        list_lengths_must_match
    ),
        extra=REMOVE_EXTRA,         # REMOVE_EXTRA = keys in data that are undefined in schema will be removed--no exception will be thrown
        required=True,              # THIS DOESN'T SEEM TO WORK: True = data is required for all defined schema keys unless overriden using
                                    #   Optional(), if not present will throw exception
)

plant_location_schema = Schema( All(
        {
        Required('latitude'): And(Coerce(float), Range(min=-90, max=90)),             # latitude [deg]
        Required('longitude'): And(Coerce(float), Range(min=-180, max=180)),          # longitude [deg]
        Required('elevation'): And(Coerce(float), Range(min=-500, max=9000)),         # elevation [m]
        Required('timezone'): And(Number(scale=0), Coerce(int), Range(min=-12, max=12))  # timezone [hr]
        },
        list_lengths_must_match
    ),
        extra=REMOVE_EXTRA,
        required=True,
)

plant_config_schema = Schema( All(
        # This is an example of validating a nested dictionary
        {
        Required('name'): Coerce(str),                                                 # plant name
        Required('location'): plant_location_schema,                                   # plant location dict
        },
        list_lengths_must_match
    ),
        extra=REMOVE_EXTRA,         # REMOVE_EXTRA = keys in data that are undefined in schema will be removed--no exception will be thrown
        required=True,              # THIS DOESN'T SEEM TO WORK: True = data is required for all defined schema keys unless overriden using
                                    #   Optional(), if not present will throw exception
)

ssc_dispatch_targets_schema = Schema( All(
        {
        Required('q_pc_max_in'): [And(Coerce(float), Range(min=0, max=kBigNumber))],
        Required('q_pc_target_on_in'): [And(Coerce(float), Range(min=0, max=kBigNumber))],
        Required('q_pc_target_su_in'): [And(Coerce(float), Range(min=0, max=kBigNumber))],
        Required('is_rec_su_allowed_in'): [And(Coerce(int), Range(min=0, max=1))],
        Required('is_rec_sb_allowed_in'): [And(Coerce(int), Range(min=0, max=1))],
        Required('is_pc_su_allowed_in'): [And(Coerce(int), Range(min=0, max=1))],
        Required('is_pc_sb_allowed_in'): [And(Coerce(int), Range(min=0, max=1))]
        },
        list_lengths_must_match
    ),
        extra=REMOVE_EXTRA,         # REMOVE_EXTRA = keys in data that are undefined in schema will be removed--no exception will be thrown
        required=True,              # THIS DOESN'T SEEM TO WORK: True = data is required for all defined schema keys unless overriden using
                                    #   Optional(), if not present will throw exception
)

dispatch_outputs_schema = Schema(
        {
        Required('current_day_schedule'): [And(Coerce(float), Range(min=0, max=kBigNumber))],
        Required('next_day_schedule'): [And(Coerce(float), Range(min=0, max=kBigNumber))],
        },
        extra=ALLOW_EXTRA,         # REMOVE_EXTRA = keys in data that are undefined in schema will be removed--no exception will be thrown
        required=True,              # THIS DOESN'T SEEM TO WORK: True = data is required for all defined schema keys unless overriden using
                                    #   Optional(), if not present will throw exception
)

plant_state_schema = Schema( All(
    {
        Required('rec_startup_time_remain_init'): And(Coerce(float), Range(min=0, max=24)),
        Required('rec_startup_energy_remain_init'): And(Coerce(float), Range(min=0, max=kBigNumber)),
        Required('T_tank_cold_init'): And(Coerce(float), Range(min=0, max=kBigNumber)),
        Required('T_tank_hot_init'): And(Coerce(float), Range(min=0, max=kBigNumber)),
        Required('pc_startup_time_remain_init'): And(Coerce(float), Range(min=0, max=24)),
        Required('pc_startup_energy_remain_initial'): And(Coerce(float), Range(min=0, max=kBigNumber)),
        Required('sf_adjust:hourly'): [And(Coerce(float), Range(min=0, max=1))]
    },
    list_lengths_must_match
),
    extra=REMOVE_EXTRA,
    # REMOVE_EXTRA = keys in data that are undefined in schema will be removed--no exception will be thrown
    required=True,
    # THIS DOESN'T SEEM TO WORK: True = data is required for all defined schema keys unless overriden using
    #   Optional(), if not present will throw exception
)
