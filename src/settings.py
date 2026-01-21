import os
from datetime import datetime

from dotenv import load_dotenv
from omegaconf import OmegaConf
from sklearn.model_selection import StratifiedKFold

from src.utils import get_env_var

load_dotenv()

# Structure: key = years_ahead, value = model_info
PRODUCTIONIZED_MODELS = {
    1: {"version_name": "flat-slide", "version_number": "44"},
    2: {"version_name": "human-assumption", "version_number": "43"},
    5: {"version_name": "warped-ribbon", "version_number": "42"},
}

DATA_DIR = "data"
MODEL_DIR = "models"
OUTPUTS_DIR = "outputs"

LOAD_DATA_FROM_AML = True
LOG_EXPERIMENT_TO_AIM = False
ANALYZE_ALGORITHM = False
PARALLELIZE = False

RANDOM_SEED = 42
CROSS_VAL_SETTING = StratifiedKFold(n_splits=5)
ALGORITHMS = ["XGBoostClassifier", "RandomForestClassifier"]
CALIBRATION_METHODS = ["no calibration", "sigmoid", "isotonic"]

# Obtain secrets
RGNAME = os.environ.get("RESOURCE_GROUP")
SUBSCRIPTIONID = os.environ.get("AML_SUBSCRIPTION_ID")
WORKSPACE_NAME = os.environ.get("AML_WORKSPACE_NAME")

# Datalake stuff
DATASTORENAME_PRD = "datalake"
DATASTORENAME_DEV = "datalakedev"
INPUT_PATH = "input/verhuiskans"
OUTPUT_SUBFOLDER_LATEST_VERHUISKANS = "output/verhuiskans/latest"
OUTPUT_SUBFOLDER_AUDITTRAIL = "output/verhuiskans/audittrail"

# Create configs for azure related task and training-related tasks.
# conf will be logged to the aim experiment tracking server, whereas azure will only be used locally.
azure = OmegaConf.create()
conf = OmegaConf.create()

azure.project_name = "verhuiskans"
azure.azureml_experiment_name = "verhuiskans-tst"

azure.application_insights_connection_string = get_env_var(
    "APPLICATION_INSIGHTS_CONNECTION_STRING", raise_if_empty=False
)

# -----------------------
# train_dates = First day of the year one year ago (i.e. near the most recent train date
# we could take for 1 year ahead predictions), plus the four years prior to this date.
conf.data = {}
# TODO: conf.data.train_dates could be rewritten to first day of the current month, so we can take
# a little more recent data to train models on. This should be done in conjunction with the TODO in
# prepare's expand_rows() function and the assert in _make_expanded_train_test_sets. Then it will be:
# conf.data.train_dates = [
#     (datetime(datetime.today().year - 1 - i, datetime.today().month, 1)).strftime("%Y-%m-%d") for i in range(5)
# ]
conf.data.train_dates = [(datetime(datetime.today().year - 1 - i, 1, 1)).strftime("%Y-%m-%d") for i in range(5)]
conf.data.test_date_years_ahead = [1, 2, 5]

# Some (or most) train_dates+years_ahead combinations are not used for production but to assess stability of
# performance over time. For each year_ahead, store the maximum train_date in a dictionary
# in conf.data.production_dates. This combination of train_date and years_ahead will ultimately be productionized.
datetime_train_dates = [datetime.strptime(date, "%Y-%m-%d") for date in conf.data.train_dates]

conf.data.production_dates = {
    y: max(
        date.strftime("%Y-%m-%d")
        for date in datetime_train_dates
        if date.replace(year=date.year + y) < datetime.today()
    )
    for y in conf.data.test_date_years_ahead
}

conf.model = {}

conf.aim_repo = f"aim://{os.environ['AIM_LOGGING_URL']}"
conf.aim_experiment = "verhuiskans"
