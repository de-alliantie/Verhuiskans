import copy
import subprocess
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from aim import Image, Run, Text
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from src.my_logging import logger
from src.settings import (
    ALGORITHMS,
    CALIBRATION_METHODS,
    CROSS_VAL_SETTING,
    OUTPUTS_DIR,
    PARALLELIZE,
    RANDOM_SEED,
    conf,
)
from src.utils.io import LEVEL, generate_data_dir_path, load_from_pkl


def train_and_evaluate_models(
    train_test_sets: dict,
    display_name: str,
    traindate_str: int,
    testdate_str: int,
    years_ahead: int,
    number_of_experiments=5,
) -> dict:
    """Train and evaluate models.

    1. For each type of algorithm, runs experiments for to find the best hyperparameters for ranking.
    2. With the model with best parameters for ranking, various calibration methods are attempted.
    3. Different evaluations are stored: cross-validated metrics, test metrics and a calibration plot.
    4. Each model is then trained on all data (train+test) and stored, should we wish to productionize it*.
    5. All models + info and calibration plot is returned

    * We have decided we wish to manually select the best model based on the calibration plot and
        ROC AUC + Brier score metrics. By storing them all, we can
        select ourselves which model to productionize.
    """
    # 0. Setting things up..
    X_train, y_train = train_test_sets["X_train"], train_test_sets["y_train"]
    X_calibrate, y_calibrate = train_test_sets["X_calibrate"], train_test_sets["y_calibrate"]
    X_test, y_test = train_test_sets["X_test"], train_test_sets["y_test"]

    results = []
    _, ax = plt.subplots(figsize=(10, 10))
    colors = plt.get_cmap("Dark2")
    color_index = 0

    # 1. Looping over ALGORITHMS
    for algorithm in ALGORITHMS:
        if algorithm == "RandomForestClassifier":
            model = RandomForestClassifier(random_state=RANDOM_SEED)
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            }
        if algorithm == "XGBoostClassifier":
            model = XGBClassifier(random_state=RANDOM_SEED)
            param_dist = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.1, 0.3],
                "subsample": [0.7, 0.8, 1.0],
                "colsample_bytree": [0.7, 0.8, 1.0],
                "gamma": [0, 0.1, 0.2],
            }

        # Randomized search optimizing for ROC AUC. Optimization for Brier score comes in the calibration step.
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=number_of_experiments,
            cv=CROSS_VAL_SETTING,
            scoring="roc_auc",
            verbose=0,
            random_state=RANDOM_SEED,
            n_jobs=1 if PARALLELIZE else -1,
        )

        search.fit(X_train, y_train)
        best_model, best_params, cv_roc_auc = search.best_estimator_, search.best_params_, search.best_score_

        # 2. Loop over all CALIBRATION_METHODS with the model best hyperparameters for ranking (= ROC_AUC)
        for calibration_method in CALIBRATION_METHODS:
            # no_calibration is also tried because there is no guarantee that using CalibratedClassifierCV yields
            # better calibrated predictions. See also: https://stackoverflow.com/q/30285551
            if calibration_method == "no calibration":
                calibrated_model = best_model
            else:
                calibrated_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(best_model),
                    method=calibration_method,
                    ensemble=False,
                )
                # Fit the calibrated model on validation data
                calibrated_model.fit(X_calibrate, y_calibrate)

            # 3. Evaluation of performance
            cv_calibrated_brier_score = brier_score_loss(
                y_true=y_calibrate, y_proba=calibrated_model.predict_proba(X_calibrate)[:, 1]
            )

            # Assess performance on test set, plot it in calibration line
            pos_class_proba = calibrated_model.predict_proba(X_test)[:, 1]

            display_kwargs = {"marker": "o", "markersize": 0.2, "linewidth": 0.3}
            CalibrationDisplay.from_predictions(
                y_true=y_test,
                y_prob=pos_class_proba,
                n_bins=40,
                strategy="quantile",
                name=f"{algorithm}_{calibration_method}_AUC_{cv_roc_auc}_Brier_{cv_calibrated_brier_score}",  # noqa: E501
                ax=ax,
                color=colors(color_index),
                **display_kwargs,
            )
            color_index += 1

            # Evaluate on test set
            test_roc_auc = round(roc_auc_score(y_true=y_test, y_score=pos_class_proba), 3)
            test_brier_score = round(brier_score_loss(y_true=y_test, y_proba=pos_class_proba), 3)

            # 4. Make model production-ready (meaning: train it on latest data)
            # The original train + calibration datasets will be used for training
            X_train_prd = np.concatenate((X_train, X_calibrate), axis=0)
            y_train_prd = np.concatenate((y_train, y_calibrate), axis=0)

            if algorithm == "RandomForestClassifier":
                best_model = RandomForestClassifier(**best_params, random_state=RANDOM_SEED)
            if algorithm == "XGBoostClassifier":
                best_model = XGBClassifier(**best_params, random_state=RANDOM_SEED)
            best_model.fit(X_train_prd, y_train_prd)

            # If valid calibration method is selected, it will be done one the original test dataset
            if calibration_method != "no calibration":
                X_calibrate_prd = X_test
                y_calibrate_prd = y_test
                calibrated_model = CalibratedClassifierCV(
                    estimator=FrozenEstimator(best_model),
                    method=calibration_method,
                    ensemble=False,
                )
                calibrated_model.fit(X_calibrate_prd, y_calibrate_prd)

            # 5. Store result to results
            results.append(
                {
                    "algorithm": algorithm,
                    "hyperparameters": best_params,
                    "calibration_method": calibration_method,
                    "cross_validated_roc_auc": cv_roc_auc,
                    "test_roc_auc": test_roc_auc,
                    "cross_validated_brier_score": cv_calibrated_brier_score,
                    "test_brier_score": test_brier_score,
                    "model": calibrated_model,
                }
            )
            # Log the results
            summary = f"""
                {display_name}
                Cross-validated results
                Algorithm: {algorithm}
                Calibration method: {calibration_method}
                CV_ROC_AUC: {cv_roc_auc}
                CV_Brier Score: {cv_calibrated_brier_score}
                Test_ROC_AUC: {test_roc_auc}
                Test_Brier Score: {test_brier_score}
            """

            # Log results to aim
            aim_run = Run(repo=conf.aim_repo, experiment=conf.aim_experiment)

            aim_run["traindate"] = traindate_str
            aim_run["testdate"] = testdate_str
            aim_run["years_ahead"] = years_ahead
            model_params = copy.deepcopy(best_params)
            model_params["type"] = algorithm
            model_params["calibration_method"] = calibration_method
            aim_run["model"] = model_params
            aim_run.track(cv_roc_auc, name="AUC", context={"subset": "train"})
            aim_run.track(test_roc_auc, name="AUC", context={"subset": "test"})
            aim_run.track(cv_calibrated_brier_score, name="Brier", context={"subset": "train"})
            aim_run.track(test_brier_score, name="Brier", context={"subset": "test"})

            ax.legend()
            plt.title(display_name)
            output_filepath = f"{OUTPUTS_DIR}/calibrationplots_{algorithm}_{calibration_method}.png"
            plt.savefig(output_filepath, dpi=300, format="png")
            aim_run.track(
                Image(output_filepath, format="png"),
                name="calibration_plot_test",
                context={"subset": "test"},
            )
            # Get current git-commit
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
            aim_run["commit_hash"] = commit_hash

            aim_run.track(Text(summary), name="summary", context={"subset": "train"})

            logger.info(summary)

    # Create calibration plot for all models in the ALGORITHMS/CALIBRATION_METHODS loops, also save it as a variable.
    ax.legend()
    plt.title(display_name)
    output_filepath = f"{OUTPUTS_DIR}/calibrationplots_{display_name.replace(' ', '_')}.png"
    plt.savefig(output_filepath, dpi=300, format="png")

    plot_buffer = BytesIO()
    plt.savefig(plot_buffer, format="png")
    plot_buffer.seek(0)  # Move to the start of the BytesIO buffer
    plt.close()

    # 6. Save results
    output = {"models": results, "calibration_plot": plot_buffer}

    return output


if __name__ == "__main__":
    train_test_path = generate_data_dir_path(LEVEL.PREPARE, "train_test_sets", suffix=".pickle")
    train_test_sets = load_from_pkl(train_test_path)
    output = train_and_evaluate_models(
        train_test_sets=train_test_sets,
        traindate_str="test",
        testdate_str="test",
        years_ahead=9999,
        display_name="JUST A TEST",
    )
