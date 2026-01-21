import mltable
import numpy as np
import pandas as pd
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from src.columns import COL_ENDDATE, COL_HOVK_STATUS, COL_STARTDATE, DATE_COLUMNS
from src.data_types import DataTypes
from src.my_logging import logger
from src.settings import (
    DATASTORENAME_PRD,
    INPUT_PATH,
    RGNAME,
    SUBSCRIPTIONID,
    WORKSPACE_NAME,
)
from src.utils.io import (
    LEVEL,
    generate_data_dir_path,
    load_df_from_csv,
    save_df_to_csv,
    save_to_pkl,
)

TOBIAS_AX_DATASET = "vhk_alle_queries_AX_backupp_v2"

DATA_ASSETS = {
    TOBIAS_AX_DATASET: {
        "version": "2",
        "type": "file",
        "filename": "vhk_alle_queries_AX_backup2.csv",
        "kwargs": {"dtype": DataTypes.all_datatypes},
    },
    "vhk_alle_queries_v2": {"version": "latest", "type": "mltable", "kwargs": {"dtype": DataTypes.all_datatypes}},
}

DEFAULT_URI = f"azureml://subscriptions/{SUBSCRIPTIONID}/resourcegroups/{RGNAME}/workspaces/{WORKSPACE_NAME}/datastores/{DATASTORENAME_PRD}/paths/{INPUT_PATH}"  # noqa:E501


def load_data_assets(for_predict: bool = False) -> None:
    """Laadt alle verhuiskans data assets uit AzureML in op basis van naam en versie, en concat deze.

    Filtert rijen met lege begin/datums eruit. Slaat ze op in raw en interim.
    """

    credential = DefaultAzureCredential()
    ml_client = MLClient(
        subscription_id=SUBSCRIPTIONID, resource_group_name=RGNAME, workspace_name=WORKSPACE_NAME, credential=credential
    )

    dfs = {}
    for data_asset_name, data_asset_details in DATA_ASSETS.items():
        if for_predict and data_asset_name == TOBIAS_AX_DATASET:
            logger.info(f"Skipping {data_asset_name} because we only require data for predictions")
            continue

        logger.info(f"loading {data_asset_name} data")

        if data_asset_details["version"] == "latest":
            data_asset_details["version"] = get_latest_data_asset_version(ml_client, data_asset_name)

        data_asset = ml_client.data.get(data_asset_name, version=int(data_asset_details["version"]))

        if data_asset_details["type"] == "mltable":
            tbl = mltable.load(data_asset.path)
            df = tbl.to_pandas_dataframe()

        elif data_asset_details["type"] == "file":
            separator = ";"

            uri = f"{DEFAULT_URI}/{data_asset_details['filename']}"
            df = pd.read_csv(uri, sep=separator, **data_asset_details["kwargs"])

        save_df_to_csv(LEVEL.LOAD, data_asset_name, df)

        df = load_df_from_csv(LEVEL.LOAD, data_asset_name, **data_asset_details["kwargs"])
        df[DATE_COLUMNS] = df[DATE_COLUMNS].apply(pd.to_datetime)
        dfs[data_asset_name] = df

    dfs_path = generate_data_dir_path(LEVEL.LOAD, "dfs_with_datatypes", suffix=".pickle")
    save_to_pkl(dfs, dfs_path)

    if for_predict:
        df_combined = dfs["vhk_alle_queries_v2"]
    else:
        df_combined = pd.concat([dfs["vhk_alle_queries_v2"], dfs[TOBIAS_AX_DATASET]], axis=0).reset_index(drop=True)

    df_combined["survival_eindjaar"] = df_combined[COL_ENDDATE].dt.year
    df_combined[COL_ENDDATE] = df_combined[COL_ENDDATE].dt.normalize()

    # Beëindigd en historisch betekent praktisch hetzelfde. Deze trekken we gelijk.
    df_combined.replace({COL_HOVK_STATUS: {"Historisch": "Beëindigd"}}, inplace=True)
    if not for_predict:
        # For train we don't want 'Opgezegd', for predict we want to keep it: it's highly informative
        df_combined = df_combined.query("huurovereenkomst_statusnaam != 'Opgezegd'")

    datakwaliteitscontrole(df_combined)
    df_combined_path = generate_data_dir_path(LEVEL.LOAD, "df_combined", suffix=".pickle")
    save_to_pkl(df_combined, df_combined_path)
    return None


def get_latest_data_asset_version(ml_client, data_asset_name) -> str:
    """Gets a list of data assets, and returns the latest version."""
    versions = []
    data_assets = ml_client.data.list(name=data_asset_name)
    for data_asset in data_assets:
        versions.append(int(data_asset.version))

    # Sort in ascending order
    sorted_versions = sorted(versions)

    return str(sorted_versions[-1])


def datakwaliteitscontrole(df: pd.DataFrame) -> None:
    """Loopt datakwaliteitsregels na en raist een error wanneer hieraan niet is voldaan.

    Args:
        df (pd.DataFrame): te controleren dataframe
    """
    einddatum_niet_leeg = (~np.isnat(df[COL_ENDDATE])).all()
    startdatum_niet_leeg = (~np.isnat(df[COL_STARTDATE])).all()

    assert einddatum_niet_leeg, "Lege einddatum gevonden!"
    assert startdatum_niet_leeg, "Lege startdatum gevonden!"
    return None


if __name__ == "__main__":
    load_data_assets()
