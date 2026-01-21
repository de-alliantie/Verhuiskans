from datetime import datetime

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.columns import (
    CAT_COLUMNS,
    COL_ENDDATE,
    COL_HOVK_STATUS,
    COL_ID_EENHEID,
    COL_ID_HOVK,
    COL_LABEL_DURATION,
    COL_LABEL_EVENT,
    COL_STARTDATE,
    FEATURE_COLUMNS,
    NUM_COLUMNS,
)
from src.utils.io import LEVEL, generate_data_dir_path, load_from_pkl, save_to_pkl

FRAC = 1  # percentage van de data die je meeneemt (voor testen, zet bijv. op 0.01)


class DataPreprocessor:
    """Class to handle all data preparations."""

    def __init__(self, traindate: datetime, testdate: datetime, years_ahead: int, expand_interval: int = 1):
        """Initializes the class."""
        self.traindate = traindate
        self.testdate = testdate
        self.years_ahead = years_ahead
        self.expand_interval = expand_interval

        # Placeholder for variables to assign while preparing
        self.df = None
        self.trainset = None
        self.testset = None
        self.pipe = None
        self.train_test_sets = None
        self.testset_timewindow = None

    def __call__(self) -> tuple[dict | None, ColumnTransformer | None]:
        """Calls the prepare function and returns train_test_sets and preprocessing pipeline."""
        self.prepare()
        return self.train_test_sets, self.pipe

    def prepare(self) -> None:
        """General data preparation function.

        Returns:
            dict: containing X and y for train-test-sets
            pipe: fitted preprocessing pipeline
        """
        df_combined_path = generate_data_dir_path(LEVEL.LOAD, "df_combined", suffix=".pickle")
        self.df = load_from_pkl(df_combined_path)
        # We zetten actieve huurovereenkomsten op pd.NaT i.p.v. 2199-12-31
        self.df.loc[self.df[COL_HOVK_STATUS] == "Actief", COL_ENDDATE] = pd.NaT

        self._vervang_lege_waardes_met_dummies()
        self.df_original = self.df.copy()

        # Sampling percentage van originele df (bijv. voor testen)
        self.df = self.df.sample(frac=FRAC, random_state=1)

        # Generate multiple peildatums
        self._expand_rows()

        # Create peildatum based variables
        self.df = create_peildatum_based_variables(df=self.df, years_ahead=self.years_ahead)

        self._make_expanded_train_test_sets()

        train_test_path = generate_data_dir_path(LEVEL.PREPARE, "train_test_sets", suffix=".pickle")
        save_to_pkl(self.train_test_sets, train_test_path)

        return None

    def _vervang_lege_waardes_met_dummies(self) -> None:
        """Vervang lege waarden bij bk_huurovereenkomst met "H1", "H2", ..., "HN" en doe hetzelfde bij bk_eenheid.

        Deze komen uit Tobias AX en zijn leeg vanwege privacy.
        """
        empty_indices = self.df.bk_huurovereenkomst.isna()
        eenheidcode_replacements = [f"E{i + 1}" for i in range(empty_indices.sum())]
        hovkcode_replacements = [f"H{i + 1}" for i in range(empty_indices.sum())]
        self.df.loc[empty_indices, COL_ID_HOVK] = hovkcode_replacements
        self.df.loc[empty_indices, COL_ID_EENHEID] = eenheidcode_replacements
        return None

    def _expand_rows(self) -> None:
        """Expand rows for every peildatum between COL_STARTDATE up until traindate, every self.expand_interval years.

        This is including peildatum = traindate, because we use this peildatum to extract the testset. We also expand
        for peildatums on the first day of every month in the 12 months leading up to the einddatum huurcontract, if the
        contract is currently terminated. Note that in this case we duplicate one contract into 12 data points that have
        y=1 and a variable number of data points with y=0. It seems that this methodology produces a good balance
        between y=1 and y=0 data points, since the resulting model predicts ones and zeros in the right proportion.
        """
        # Obtain global minimum startdate, generate sequence
        global_startdate = self.df[COL_STARTDATE].min()
        global_annual_dates = self._create_date_sequence(startdate=global_startdate)

        # Take product of COL_ID_HOVK and peildatum, expand these and merge with self.df
        multi_index = pd.MultiIndex.from_product(
            [self.df[COL_ID_HOVK], global_annual_dates], names=[COL_ID_HOVK, "peildatum"]
        )
        expanded_df = pd.DataFrame(index=multi_index).reset_index()

        expanded_df = expanded_df.merge(self.df, on=COL_ID_HOVK)

        # Filter out combinations where peildatum is before COL_STARTDATE and after COL_ENDDATE
        expanded_df = expanded_df[expanded_df["peildatum"] > expanded_df[COL_STARTDATE]]
        expanded_df = expanded_df.loc[lambda x: (x["peildatum"] < x[COL_ENDDATE]) | pd.isna(x[COL_ENDDATE])]

        expanded_df[COL_ENDDATE] = expanded_df[COL_ENDDATE].dt.normalize()
        # TODO: the code below could be rewritten to be in sync with the TODO in settings.conf.data.train_dates
        # when we don't require first day of the year but first day of the current month, we can take a little more
        # recent data to train models on.
        expanded_df["is_1_januari"] = (expanded_df["peildatum"].dt.month == 1) & (expanded_df["peildatum"].dt.day == 1)

        # Take peildatum every year (is_1_januari) AND every month of last 12 months of huurovereenkomst
        df1 = expanded_df.loc[lambda x: ((x[COL_ENDDATE] - x["peildatum"]).dt.days <= 365) | x["is_1_januari"]]

        self.df = df1

        return None

    def _create_date_sequence(self, startdate: datetime):
        """Create a range of annual dates between startdate of rental agreement and testdate.

        Args:
            startdate (datetime): global_startdate

        Returns:
            list: list of first-day-of-years between global startdate and testdate
        """

        start_year = startdate.year + 1  # start_year +1 om negatieve huurduur te voorkomen
        end_year = self.traindate.year + 1  # end_year +1 om het eindjaar in de range te houden
        years = range(start_year, end_year, self.expand_interval)
        months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
        dates = [pd.to_datetime(f"{year}-{month}-01") for year in years for month in months]
        return dates

    def _make_expanded_train_test_sets(self) -> None:
        """Makes time-related train and test sets. Pushes the sets through data processing pipeline. Stores them in a
        dict.

        Train set: all data with peildatum until (but not including!) years_ahead before the traindate.
        The following example explains this construction. Assume years_ahead=1 and traindate = 1-1-2020
        Then we cannot use data points with peildatum in 2019 to include in the trainset, since if a
        contract is still active at the traindate, we don't know its label yet. For example if
        peildatum=1-11-2019 and the contract enddate is 1-3-2020, we don't know at the traindate
        that y=1 yet. If it had been terminated between peildatum and traindate, you still cannot include it
        since you will create a bias.
        This dataset is used to learn a machine learning model to rank all contracts from low chance of
        churning to high chance of churning.

        Calibration set:The peildatum years_ahead before the traindate is used as a calibration set.
        This set is used to create better calibrated probabilities. Expanding the training dataset will enable the
        machine learning model to better learn the relationship between X and y, but it has artificially created
        a lot of COL_LABEL_EVENT==True (approx 55% of instances). In reality this is about 5% of instances.
        The calibration set has a more realistic distribution and as such helps to calibrate y-pred closer to y-true.

        Test set: all data with peildatum at the traindate. Do note that if we "peil" a contract,
        it is still active at that moment! So these are active contracts at the traindate. We can test on this set
        if we can predict whether the contract mutates between traindate and traindate + years_ahead. The latter
        has to be in the past so that we know the outcome.

        Note: COL_LABEL_EVENT and COL_LABEL_DURATION have already been calculated based on peildatum.
        Note: COL_LABEL_DURATION is part of X to allow algorithm to learn relationship
        between huurduur and huuropzegging
        """
        # Hovks that started after traindate will not be used for training & testing
        self.df = self.df.loc[lambda x: x[COL_STARTDATE] <= self.traindate]

        self.trainset = self.df[self.df["peildatum"] <= self.traindate - pd.DateOffset(years=self.years_ahead)]

        # Onderstaande regel is hoe we omgaan met survivorship bias. Zie Confluence voor gemaakte keuzes.
        self.trainset = self.trainset.query("startjaar_huurovereenkomst >= 2002")

        # Laatste peildatum in trainset wordt calibratieset & verwijderd uit trainset
        max_peildatum = self.trainset.peildatum.max()
        # TODO: rewrite this assert together with TODO's in settings.conf.data.train_dates and prepare.expand_rows.
        assert (
            max_peildatum.day == 1 and max_peildatum.month == 1
        ), "Calibration set peildatum is not the 1st of January and will have a skewed distribution of COL_LABEL_EVENT!"
        self.calibratieset = self.trainset[self.trainset.peildatum == max_peildatum]
        self.trainset = self.trainset[self.trainset.peildatum != max_peildatum]

        # The testset is the dataset at peildatum==traindate, see the docstring above.
        peildatum_is_traindate = self.df["peildatum"] == self.traindate
        testset = self.df[peildatum_is_traindate]
        self.testset = testset

        X_cols = FEATURE_COLUMNS

        self._get_preprocessing_pipeline()
        X_train = self.pipe.fit_transform(self.trainset[X_cols])

        X_train = pd.DataFrame(X_train, columns=self.pipe.get_feature_names_out())
        X_calibrate = pd.DataFrame(
            self.pipe.transform(self.calibratieset[X_cols]), columns=self.pipe.get_feature_names_out()
        )
        X_test = pd.DataFrame(self.pipe.transform(self.testset[X_cols]), columns=self.pipe.get_feature_names_out())

        y_train = self.trainset[COL_LABEL_EVENT]
        y_calibrate = self.calibratieset[COL_LABEL_EVENT]
        y_test = self.testset[COL_LABEL_EVENT]

        self.train_test_sets = {
            "X_train": X_train,
            "X_calibrate": X_calibrate,
            "X_test": X_test,
            "y_train": y_train,
            "y_calibrate": y_calibrate,
            "y_test": y_test,
        }

        return None

    def _get_preprocessing_pipeline(self) -> None:
        """Scales and imputes missing values for numerical columns.

        Imputes missing values and one-hot-encodes (OHE) categorical columns. Retains max. 10 most frequently occurring
        OHE-categories minus first category to prevent multicollinearity issues. See for more details:
        https://github.com/scikit-learn/scikit-learn/issues/23436
        """
        cat_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "one-hot-encoder",
                    OneHotEncoder(handle_unknown="ignore", max_categories=10, drop="first"),
                ),
            ]
        )

        num_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

        self.pipe = ColumnTransformer(
            [
                ("categorical", cat_pipeline, CAT_COLUMNS),
                ("numerical", num_pipeline, NUM_COLUMNS),
            ]
        )
        return None


def create_peildatum_based_variables(df: pd.DataFrame, years_ahead: int) -> pd.DataFrame:
    """Creates new variables out of date columns that are calculated with peildatum.

    This function is not part of the DataPrerocessor class as it is used for both training and prediction tasks.
    """
    df1 = df.copy()

    # COL_LABEL_EVENT & COL_LABEL_DURATION are calculated based on peildatum.
    # COL_LABEL_EVENT is Y (target) and COL_LABEL_DURATION and is part of X.
    df1[COL_LABEL_DURATION] = (df1["peildatum"] - df1[COL_STARTDATE]).dt.days

    # If the peildatum is less than self.years_ahead years before the huurcontract einddatum, the label is
    # y=1, else it is y=0. So for example if we look 1 year ahead (365 days), and if einddatum = 1-1-2016,
    # then Y=0 if we "peil" (gauge) before 1-1-2015, and Y=1 if we peil after 1-1-2015.
    df1[COL_LABEL_EVENT] = (df1[COL_ENDDATE] - df1["peildatum"]).dt.days < 365 * years_ahead

    df1.loc[:, "leeftijd_woning"] = df1["peildatum"].dt.year - df1["opleverdatum"].dt.year
    df1.loc[:, "min_leeftijd"] = df1["peildatum"].dt.year - df1["min_geboortedatum"].dt.year
    df1.loc[:, "max_leeftijd"] = df1["peildatum"].dt.year - df1["max_geboortedatum"].dt.year

    return df1


if __name__ == "__main__":

    # Example usage
    traindate = "2019-01-01"
    years_ahead = 1
    traindate = pd.to_datetime(traindate)
    testdate = traindate + pd.offsets.DateOffset(years=years_ahead)

    preprocessor = DataPreprocessor(traindate=traindate, testdate=testdate, years_ahead=years_ahead)
    train_test_sets, pipeline = preprocessor()
    train_test_path = generate_data_dir_path(LEVEL.PREPARE, "train_test_sets", suffix=".pickle")
    save_to_pkl(train_test_sets, train_test_path)
