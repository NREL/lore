# Test case study.
# To run this file: python -m pytest mediation/test_single_run_casescpy
import pytest
import pathlib
import datetime
import pytz
import pandas as pd

PARENT_DIR = str(pathlib.Path(__file__).parents[1])

from mediation import mediator
from mediation.models import TechData

@pytest.mark.django_db
def test_single_run_case():
    m = mediator.Mediator(
        params_path=PARENT_DIR + "/tests/inputs/mediator_settings_test.json",
        plant_design_path=PARENT_DIR + "/tests/inputs/plant_design_test.json",
        weather_file=PARENT_DIR + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv",
        dispatch_params_path=PARENT_DIR + "/tests/inputs/dispatch_settings_test.json",
        update_interval=datetime.timedelta(seconds=5),
    )
    m.run_once(
        datetime_start=datetime.datetime(2021, 6, 1, 0, 0, 0, tzinfo=pytz.UTC),
        datetime_end=datetime.datetime(2021, 6, 3, 0, 0, 0, tzinfo=pytz.UTC),
    )
    queryset = TechData.objects.values_list('W_grid_no_derate', 'E_tes_charged')
    df = pd.DataFrame.from_records(queryset)
    # Test the net output energy [MWh-e]
    assert(round(df[0].values.sum() / 12000, 0)  == 1113)
    # test the avg. TES SOC [MWh-t]
    assert(round(df[1].values.sum() / (1000*len(df[1].values)), 0) == 1266)
    return

@pytest.mark.django_db
def test_comparison_plot():
    import os
    try:
        os.remove("./dispatch_plots.png")
    except FileNotFoundError:
        pass
    m = mediator.Mediator(
        params_path=PARENT_DIR + "/tests/inputs/mediator_settings_plots.json",
        plant_design_path=PARENT_DIR + "/tests/inputs/plant_design_test.json",
        weather_file=PARENT_DIR + "/data/daggett_ca_34.865371_-116.783023_psmv3_60_tmy.csv",
        dispatch_params_path=PARENT_DIR + "/tests/inputs/dispatch_settings_test.json",
        update_interval=datetime.timedelta(seconds=5),
    )
    m.run_once(
        datetime_start=datetime.datetime(2021, 6, 1, 0, 0, 0, tzinfo=pytz.UTC),
        datetime_end=datetime.datetime(2021, 6, 3, 0, 0, 0, tzinfo=pytz.UTC),
    )
    # Test the existence of the plots file
    assert(os.path.exists("./dispatch_plots.png"))
    os.remove("./dispatch_plots.png")
    return
