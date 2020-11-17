# from django.test import TestCase
from voluptuous import Schema, Required, Optional, Range, And, Or, DefaultTo, SetTo, Any, Coerce, Maybe, ALLOW_EXTRA, All
import random

# Create your tests here.
kBigNumber = 1.0e10
kEpsilon = 1.0e-10

if __name__ == "__main__":
    """
    Looking to:
    # - verify within a min and max
    # - convert values (e.g., MW to kW)
    # - set a default value if key is missing (i.e., None)
    # - change invalid values to another (i.e., None)

    Background:
    # - Cerebus has the best and most comprehensive features and syntax, but is unbelievably slow.
    #   It doesn't appear speed is an issue with any other validators, see: https://validx.readthedocs.io/en/latest/benchmarks.html
    #   However, it makes me hesistant to create my own validator.
    # - Some other validators have similarly good syntax, but lack the needed features.
    # - Voluptuous has all the needed features, is relatively fast, second biggest community, very flexible (even more so than Cerebus) but
    #   has a more difficult to read programming syntax compared to a markup-like one

    Random Notes:
    # Clamp() sets an out-of-bounds value to the nearest limit
    """

    test_data = {
        'none_value': [None],
        'original_name_kW': [-1, 4, -3, None],
        'original_name_variation_MW': [-1, 3, -2, None],
        # 'new_name': [-1, 5, -2, None],
        'unique_name_MW': [-2, -1, 5, None],
        'lots_of_values_MW': random.sample(range(10, 30000), k=2000)
    }

    # Normalization (coercion) functions
    def MwToKw(mw):
        try:
            mw = float(mw)
        except:
            return None
        else:
            return mw * 1.e3


    schema = Schema( All(
        # First pass for renaming and normalization
        {
        # Explanation: Since both variables are present and are renamed to the same key, the latter entry IN TEST_DATA overwrites the first one
        Optional(And('original_name_variation_MW', SetTo('new_name'))): [Coerce(MwToKw)],
        Optional(And('original_name_kW', SetTo('new_name'))): object,                           # 'object' means accept value as-is
        },

        # Second pass for validation
        {
        # Explanation:
        # 1. Require the key (e.g., 'missing_name') be present. If not present, set to the one-entry list: [None]
        # 2. Specify the key's value is a list by enclosing conditions in brackets
        # 3. For each list value, try to coerce to a float and then verify it is within a min/max range. If either throws exception, set value to None.
        Required('missing_name', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        Required('none_value', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        Required('new_name', default=[None]): [Any(And(Coerce(float), Range(min=0, max=kBigNumber)), SetTo(None))],
        # You need the 'float' in the following because Range() can't be passed a None value. 'Coerce(float)' from above, and 'float' here both throw an
        # exception if given a None, which is what you want in order to move on to the second Any() parameter, which is 'SetTo(None)'
        Required('unique_name_MW', default=[None]): [Any(And(Coerce(MwToKw), float, Range(min=0, max=kBigNumber)), SetTo(None))],
        Required('lots_of_values_MW', default=[None]): [Any(And(Coerce(MwToKw), float, Range(min=0, max=kBigNumber)), SetTo(None))],
        }
    ),
        extra=ALLOW_EXTRA,          # ALLOW_EXTRA = undefined keys in data won't cause exception
                                    # REMOVE_EXTRA = undefined keys in data will be removed
        required=False              # False = data without defined keys in schema will be kept
    )


    results = schema(test_data)

    pass