from django.conf import settings

from mediator import nested_inputs
from voluptuous import Schema, Required, Optional, Range, And, Or, DefaultTo, SetTo, Any, Coerce, Maybe, ALLOW_EXTRA, All

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

    Notes:
    - See documentation: https://alecthomas.github.io/voluptuous/docs/_build/html/voluptuous.html
    - Designated invalid values for the candidate databases:
        PostgreSQL = NaN or NULL    # NaN is a special numerical value, NULL is a missing value
        SQLite3 = NULL              # Use NULL, as NaN are converted to NULL anyway
                                    #  (NULLs may be returned as 0's, so always type checking is recommended)
    - Python does not have NULL, rather it uses the None object.
    - Python NaN is created by float("NaN") and can be tested for using math.isnan()
    """

    update = settings.DEBUG    # don't let missing required variables cause a failure while debugging

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

pysam_schema = Schema( All(
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
        {
        Required('defocus', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1.)), SetTo(None))],                     # Field optical focus fraction [-]
        Required('disp_pceff_expected', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1.)), SetTo(None))],         # Dispatch expected power cycle efficiency adj. [-]
        Required('disp_tes_expected', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],   # Dispatch expected TES charge level [MWht]
        Required('disp_wpb_expected', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],   # Dispatch expected power generation [MWe]
        Required('e_ch_tes', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],            # TES charge state [MWht]
        Required('helio_positions', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],     # Heliostat position table [matrix]
        Required('P_cycle', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],             # PC electrical power output, gross [MWe]
        Required('P_out_net', default=[None]): [Any(And(Coerce(float), Range(min=-kBigNumber, max=kBigNumber)), SetTo(None))],  # Total electric power to grid [MWe]
        Required('P_rec_heattrace', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],     # Receiver heat trace parasitic load [MWe]
        Required('q_dot_rec_inc', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],       # Receiver incident thermal power [MWt]
        Required('q_pc_startup', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],        # PC startup thermal energy [MWht]
        Required('q_sf_inc', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],            # Field incident thermal power [MWt]
        Required('q_startup', default=[None]): [Any(And(Coerce(float), Range(min=0., max=kBigNumber)), SetTo(None))],           # Receiver startup thermal energy consumed [MWt]
        Required('Q_thermal', default=[None]): [Any(And(Coerce(float), Range(min=-kBigNumber, max=kBigNumber)), SetTo(None))],  # Receiver thermal power to HTF less piping loss [MWt]
        Required('sf_adjust_out', default=[None]): [Any(And(Coerce(float), Range(min=0., max=1.)), SetTo(None))],               # Field availability adjustment factor [-]
        Required('T_pc_in', default=[None]): [Any(And(Coerce(float), Range(min=-50., max=1000.)), SetTo(None))],                # PC HTF inlet temperature [C]
        },
    ),
        extra=ALLOW_EXTRA,          # ALLOW_EXTRA = undefined keys in data won't cause exception
                                    # REMOVE_EXTRA = undefined keys in data will be removed
        required=False              # False = data without defined keys in schema will be kept
    )