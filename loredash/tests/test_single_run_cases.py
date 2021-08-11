# Test case study.
# To run this file: python -m pytest mediation/test_mediator.py
import pytest
import pathlib
import datetime
import pytz

PARENT_DIR = str(pathlib.Path(__file__).parents[1])

from mediation import mediator

@pytest.mark.django_db
def test_single_run_case():
    plant_design_path = PARENT_DIR + "/../loredash/config/plant_design.json"
    mediator_params_path = PARENT_DIR + "/config/mediator_settings.json"
    dispatch_params_path = PARENT_DIR + "/config/dispatch_settings.json"
    m = mediator.Mediator(
        params_path = mediator_params_path,
        plant_design_path=plant_design_path,
        weather_file = PARENT_DIR + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv",
        dispatch_params_path=dispatch_params_path,
        update_interval = datetime.timedelta(seconds = 5),
    )
    tzinfo = pytz.UTC
    datetime_start = datetime.datetime(2021, 1, 1, 0, 0, 0, tzinfo=tzinfo)
    datetime_end = datetime.datetime(2021, 1, 3, 0, 0, 0, tzinfo=tzinfo)
    m.run_once(datetime_start=datetime_start,
                         datetime_end=datetime_end)

if __name__ == "__main__":
    test_single_run_case()
