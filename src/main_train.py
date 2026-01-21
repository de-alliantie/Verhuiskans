import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import randomname

from src.load import load_data_assets
from src.my_logging import logger
from src.prepare import DataPreprocessor
from src.settings import (
    ALGORITHMS,
    CALIBRATION_METHODS,
    LOAD_DATA_FROM_AML,
    MODEL_DIR,
    PARALLELIZE,
    azure,
    conf,
)
from src.train import train_and_evaluate_models
from src.utils.aml_models import upload_model_to_AML
from src.utils.io import load_from_pkl, save_to_pkl

pd.set_option("future.no_silent_downcasting", True)


def run_train_jobs():
    """Run all training runs of train_dates + years_ahead.

    Neither of these options is significantly faster than the other:
    - If PARALLELIZE, all train jobs are run concurrently. This disables parallel
        processing of cross-validation in train. Still it's a bit quicker.
    - If not PARALLELIZE, even though you run train jobs consecutively, cross-validation
        in train is done in parallel.
    """
    exp_name = azure.project_name
    logger.info(f"Experiment name: {exp_name}")

    if LOAD_DATA_FROM_AML:
        load_data_assets()

    traindates = conf.data.train_dates
    years_ahead_list = conf.data.test_date_years_ahead

    processes = [(traindate, years_ahead) for traindate in traindates for years_ahead in years_ahead_list]
    processes = sorted(processes, key=lambda x: (x[1], x[0]))

    if PARALLELIZE:
        logger.warning(
            "You are parallelizing the train runs. Ensure you run it from a compute with sufficient cores and RAM."
        )
        with Pool() as pool:
            pool.map(_train_pipeline, processes)  # Parallel execution
    else:
        for train_job in processes:
            _train_pipeline(train_job)


def _train_pipeline(args: tuple[str, int]) -> None:
    """Main function for loading-preprocessing-training of a given traindate + years_ahead.

    - it prepares the data into temporal train/test splits
    - it trains models on the data
    - it evaluates the model on the test set
    - it saves the outputs if it's a run that might be productionized (see settings.conf.data.production_dates)
    """
    traindate, years_ahead = args
    traindate = pd.to_datetime(traindate)
    testdate = traindate + pd.offsets.DateOffset(years=years_ahead)
    traindate_str = str(traindate)[:10]
    testdate_str = str(testdate)[:10]

    basic_logging = f"TRAINDATE {traindate_str} TESTDATE {testdate_str} YA {years_ahead}:"

    if testdate > datetime.today():
        logger.info(f"{basic_logging} Testdate is in the future and therefore skipped.")
        return None

    logger.info(f"{basic_logging} Preparing data..")
    preprocessor = DataPreprocessor(traindate=traindate, testdate=testdate, years_ahead=years_ahead)
    train_test_sets, pipeline = preprocessor()

    logger.info(f"{basic_logging} Training model..")
    model_dict = train_and_evaluate_models(
        train_test_sets=train_test_sets,
        display_name=basic_logging,
        traindate_str=traindate_str,
        testdate_str=testdate_str,
        years_ahead=years_ahead,
    )

    model_dict["train_test_sets"] = train_test_sets
    model_dict["pipeline"] = pipeline
    model_dict["traindate"] = traindate
    model_dict["testdate"] = testdate
    model_dict["years_ahead"] = years_ahead

    if conf.data.production_dates[years_ahead] == traindate_str:
        logger.info(f"{basic_logging} Saving potential models to productionize to {MODEL_DIR}/.")
        os.makedirs(f"./{MODEL_DIR}/trained/", exist_ok=True)
        model_path = Path(
            f"./{MODEL_DIR}/trained/{azure.project_name}_traindate_{traindate_str}_testdate_{testdate_str}.pickle"
        )
        save_to_pkl(model_dict, model_path)
    else:
        logger.info(f"{basic_logging} Run was only to assess stability over time, models are not saved.")

    return None


def pick_model_to_productionize() -> None:
    """After train jobs have run, this function guides you through selecting the models to productionize.

    - Finds all models in MODEL_DIR
    - Saves a calibration plot for you to select your preferred algorithm+calibration method
    - Asks you in the terminal to confirm your chosen model
    - Saves this model locally and uploads it to Azure ML with tags and properties
    """
    trained_models_path = f"{MODEL_DIR}/trained"
    for path in os.listdir(trained_models_path):
        if not path.endswith(".pickle"):
            continue
        full_path = os.path.join(trained_models_path, path)

        # Get model dictionary
        model_dict = load_from_pkl(full_path)
        traindate = model_dict["traindate"].strftime("%Y%m%d")
        testdate = model_dict["testdate"].strftime("%Y%m%d")
        years_ahead = model_dict["years_ahead"]

        # Output calibration plot for you to select preferred algorithm+calibration method for production.
        calibration_plot = model_dict["calibration_plot"]
        calibration_plot.seek(0)
        os.makedirs("calibrationplots", exist_ok=True)
        calibration_plot_path = Path(f"calibrationplots/traindate_{traindate}_{years_ahead}_years_ahead.png")
        with open(calibration_plot_path, "wb") as new_file:
            new_file.write(calibration_plot.read())

        logger.info(f"Please inspect calibration plot at: {calibration_plot_path}...")

        # Enter and confirm your choice
        while True:
            algorithm = input(f"Which algorithm to productionize? {ALGORITHMS}:")
            if algorithm not in ALGORITHMS:
                logger.info(f"Your selected algorithm '{algorithm}' is not in {ALGORITHMS}. Try again.")
                continue
            calibration_method = input(
                f"Which calibration method of {algorithm} to productionize? {CALIBRATION_METHODS}:"
            )
            if calibration_method not in CALIBRATION_METHODS:
                logger.info(
                    f"Your selected calibration method '{calibration_method}'is not in {CALIBRATION_METHODS}. Try again."  # noqa: E501
                )
                continue
            confirm = input(
                f"You have selected algorithm: {algorithm} with calibration method: {calibration_method}, Proceed? (y/n):"  # noqa: E501
            )  # noqa: E501
            if confirm != "y":
                continue
            break

        # Pick up selected algorithm+calibration method, save it locally
        selected_model_dict = next(
            model
            for model in model_dict["models"]
            if model["algorithm"] == algorithm and model["calibration_method"] == calibration_method
        )

        to_upload = {"model": selected_model_dict["model"], "pipeline": model_dict["pipeline"]}
        algorithm_name = azure.project_name
        version_name = randomname.get_name()
        model_name = f"{algorithm_name}_{version_name}_{years_ahead}_years_ahead_trained_until_{traindate}"
        logger.info(
            f"Locally saving the model trained until {traindate} that predicts {years_ahead} year(s) ahead as {model_name}."  # noqa: E501
        )

        model_path = Path(f"{MODEL_DIR}/to_aml/{model_name}.pickle")
        save_to_pkl(to_upload, model_path)

        # Extract tags and properties from model dictionary and upload to AML
        tags = {
            "version_name": version_name,
            "algorithm": selected_model_dict["algorithm"],
            "calibration_method": selected_model_dict["calibration_method"],
            "traindate": traindate,
            "testdate": testdate,
            "years_ahead": years_ahead,
        }
        properties = {
            "PARAMETERS": selected_model_dict["hyperparameters"],
            "CV_ROC_AUC": selected_model_dict["cross_validated_roc_auc"],
            "TEST_ROC_AUC": selected_model_dict["test_roc_auc"],
            "CV_BRIER_SCORE": selected_model_dict["cross_validated_brier_score"],
            "TEST_BRIER_SCORE": selected_model_dict["test_brier_score"],
        }
        upload_model_to_AML(model_path=model_path, tags=tags, properties=properties)
        logger.info(f"Uploaded the model {version_name} to Azure ML!")

    return None


if __name__ == "__main__":
    run_train_jobs()
    pick_model_to_productionize()
