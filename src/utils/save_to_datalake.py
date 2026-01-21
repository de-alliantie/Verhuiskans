import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from azureml.core import Workspace
from azureml.core.authentication import MsiAuthentication
from azureml.fsspec import AzureMachineLearningFileSystem

from src.my_logging import logger
from src.settings import (
    DATASTORENAME_DEV,
    DATASTORENAME_PRD,
    OUTPUT_SUBFOLDER_AUDITTRAIL,
    OUTPUT_SUBFOLDER_LATEST_VERHUISKANS,
    OUTPUTS_DIR,
    RGNAME,
    SUBSCRIPTIONID,
    WORKSPACE_NAME,
)


def save_outputs_to_datalake(df: pd.DataFrame) -> None:
    """Saves outputs both as latest parquet and timestamped in an audittrail folder."""

    _save_and_upload_as_parquet(
        output=df,
        local_path=OUTPUTS_DIR,
        filename="latest_verhuiskansen",
        datalake_path=OUTPUT_SUBFOLDER_LATEST_VERHUISKANS,
        env="prd" if os.environ.get("OTAP") == "P" else "dev",
    )

    _save_and_upload_as_parquet(
        output=df,
        local_path=OUTPUTS_DIR,
        filename=f"""{datetime.now().strftime("%Y%m%d")}_verhuiskansen""",
        datalake_path=OUTPUT_SUBFOLDER_AUDITTRAIL,
        env="prd" if os.environ.get("OTAP") == "P" else "dev",
    )

    return None


def _save_and_upload_as_parquet(
    output: pd.DataFrame, local_path: str, filename: str, datalake_path: str, env: str
) -> None:
    """Saves file as parquet and uploads it to datalake."""
    # Save df locally as parquet
    filename = f"{filename}.parquet"

    logger.info(f"Saving {filename} in folder {local_path}")
    Path(local_path).mkdir(parents=True, exist_ok=True)

    output.to_parquet(
        path=Path(local_path) / filename,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        storage_options=None,
    )

    logger.info(f"Uploading {filename} to {env} datalake folder {datalake_path}")

    fs = _fs_helper(datalake_path, env)
    fs.upload(
        lpath=str(Path(local_path) / filename),
        rpath=datalake_path,
        recursive=False,
        **{"overwrite": "MERGE_WITH_OVERWRITE"},
    )
    return None


def _fs_helper(paths: str, environment: str = "dev") -> AzureMachineLearningFileSystem:
    """Helper to get filesystem in Datastores.

    If environment is 'prd', the production datastore is used, otherwise the development datastore is used.
    """
    datastorename = DATASTORENAME_PRD if environment == "prd" else DATASTORENAME_DEV

    try:
        # If using an Azure Managed identity, a workspace needs to be initialized with it
        # prior to initializing the AzureMachineLearningFileSystem
        msi_auth = MsiAuthentication(identity_config={"client_id": os.getenv("AZURE_CLIENT_ID")})
        Workspace(
            subscription_id=SUBSCRIPTIONID,
            resource_group=RGNAME,
            workspace_name=WORKSPACE_NAME,
            auth=msi_auth,
        )
    except Exception:
        logger.warning(
            "Failed to initialize a Workspace with Azure Client ID.\n"
            "Attempting to initialize AzureMachineLearningFileSystem without it.\n"
            "You may ignore this warning if you are developing locally."
        )

    return AzureMachineLearningFileSystem(
        f"azureml://subscriptions/{SUBSCRIPTIONID}/resourcegroups/{RGNAME}/workspaces/{WORKSPACE_NAME}/datastores/{datastorename}/paths/{paths}"  # noqa: E501
    )
