from django.conf import settings

from mediator import nested_inputs
from cerberus import Validator

def validate(values, schema, **kwargs):
    """
    Checks a dictionary of values against a validation schema within another dict

    values      one-level dictionary of values
    schema      one-level dictionary of validation criteria with keys:
                    "type": "int",
                    "min": 1,
                    "max": 8760,
                    "required": True,
                    "description": "number of time periods"
    **kwargs    other named arguments for the validate() method

    Returns 'None' if the validation failed or the validated values if the validation succeeds. The validation fails if:
        - There is a value not of the specified type
        - There is a value not in the specified min/max range
        - < What if there is a provided value not in the schema dictionary? Fail? Keep? Remove? See 'allow_unknown' and 'purge_unknown'.>
        - < What if there is a missing value that is specified in the schema but has no default? Fail? See 'update'.>
        - < What if a value is NaN or None? Fail? Keep? See 'ignore_none_values'.>
        - < What if a value is required and it is a None? Fail? Replace with default if present? >
    If the validation succeeds, it returns:
        - All of the values given
        - Respective default values for missing values if they are specified

    Notes:
    - See documentation: https://docs.python-cerberus.org/en/stable/validation-rules.html
    - Designated invalid values for the candidate databases:
        PostgreSQL = NaN or NULL    # NaN is a special numerical value, NULL is a missing value
        SQLite3 = NULL              # Use NULL, as NaN are converted to NULL anyway
                                    #  (NULLs may be returned as 0's, so always type checking is recommended)
    - Python does not have NULL, rather it uses the None object.
    - Python NaN is created by float("NaN") and can be tested for using math.isnan()
    """
    update = settings.DEBUG    # don't let missing required variables cause a failure while debugging

    v = Validator(schema)
    result = v.validated(values, update=update, **kwargs)
    if result is not None:
        sorted(result)      # sort keys alphabetically
    errors = v.errors
    return result
