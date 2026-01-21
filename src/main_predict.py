from datetime import datetime

import pandas as pd

from src.columns import COL_HOVK_STATUS, COL_ID_EENHEID, COL_ID_HOVK, FEATURE_COLUMNS
from src.load import load_data_assets
from src.my_logging import logger
from src.prepare import create_peildatum_based_variables
from src.settings import PRODUCTIONIZED_MODELS
from src.utils.aml_models import get_model_from_AML
from src.utils.io import LEVEL, generate_data_dir_path, load_from_pkl
from src.utils.msteams import log_result_to_MS_teams
from src.utils.save_to_datalake import save_outputs_to_datalake


def generate_verhuiskansen() -> None:
    """Genereert verhuiskansen en slaat deze op op het datalake.

    - Haalt de meest recente data op om voorspellingen te doen
    - Loopt over PRODUCTIONIZED_MODELS, duwt recente data door pipeline en genereert verhuiskans
    - Plakt verhuiskansen van elk model onder elkaar en slaat deze op op het datalake
    """
    # Initialize outputs
    outputs = []
    # Loading in data assets is not parametrized as we assume you would always want the latest data.
    load_data_assets(for_predict=True)

    # Load in data, filter only the active huurovereenkomsten
    df_combined_path = generate_data_dir_path(LEVEL.LOAD, "df_combined", suffix=".pickle")
    df_combined = load_from_pkl(df_combined_path)
    df_combined["peildatum"] = datetime.today()

    for years_ahead, model_info in PRODUCTIONIZED_MODELS.items():
        logger.info(f"Predicting {years_ahead} year(s) ahead..")

        version_name = model_info["version_name"]
        version_number = model_info["version_number"]
        model_dict = get_model_from_AML(version_name=version_name, version_number=version_number)

        model = model_dict["model"]
        pipeline = model_dict["pipeline"]

        # Actieve huurovereenkomsten krijgen voorspelde verhuiskans
        df_actief = df_combined.loc[df_combined[COL_HOVK_STATUS] == "Actief"].copy()
        df_actief = create_peildatum_based_variables(df=df_actief, years_ahead=years_ahead)
        X = pd.DataFrame(pipeline.transform(df_actief[FEATURE_COLUMNS]), columns=pipeline.get_feature_names_out())
        predictions = pd.Series(model.predict_proba(X)[:, 1], name="verhuiskans")
        df_actief = pd.concat([df_actief[[COL_ID_HOVK, COL_ID_EENHEID]], predictions], axis=1)

        # Opgezegde huurovereenkomsten krijgen verhuiskans van 1.0
        df_opgezegd = df_combined.loc[df_combined[COL_HOVK_STATUS] == "Opgezegd", [COL_ID_HOVK, COL_ID_EENHEID]].copy()
        df_opgezegd["verhuiskans"] = 1.0

        output = pd.concat([df_actief, df_opgezegd], axis=0, ignore_index=True)
        output["aantal_jaar_vooruit"] = years_ahead
        output["voorspellingslabel"] = f"binnen {years_ahead} jaar"
        output["modelnaam"] = version_name
        output["modelversie"] = version_number
        output["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        outputs.append(output)

    outputs = pd.concat(outputs, axis=0, ignore_index=True)
    save_outputs_to_datalake(outputs)

    return None


if __name__ == "__main__":
    try:
        generate_verhuiskansen()
        message = "VERHUISKANS ALGORITME: predict run succesvol afgerond"
        logger.info(message)
        log_result_to_MS_teams(message)
    except Exception as e:
        message = f"VERHUISKANS ALGORITME: error in predict run: {e}"
        logger.info(message)
        log_result_to_MS_teams(message)
        raise
