import os
from pathlib import Path

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import DefaultAzureCredential

from src.my_logging import logger
from src.settings import MODEL_DIR, RGNAME, SUBSCRIPTIONID, WORKSPACE_NAME, azure
from src.utils.io import load_from_pkl


def upload_model_to_AML(model_path: str, tags: dict, properties: dict) -> None:
    """Deze functie registreert binnen de Model List van Azure Machine Learning:
    - het voorspellend algoritme
    - de gefitte preprocessing pipeline om te komen van ruwe features tot een inputset voor prediction
    """
    logger.info("Uploading model to Azure Machine Learning")

    # Send model_dict to AML Models section
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        subscription_id=SUBSCRIPTIONID, resource_group_name=RGNAME, workspace_name=WORKSPACE_NAME, credential=credential
    )

    file_model = Model(
        path=model_path,
        type=AssetTypes.CUSTOM_MODEL,
        tags=tags,
        properties=properties,
        name=azure.project_name,
    )
    ml_client.models.create_or_update(file_model)

    return None


def get_model_from_AML(version_name: str, version_number: str) -> object:
    """Deze functie haalt op basis van versienaam en versienummer uit de Model List van Azure Machine Learning:
    - het voorspellend algoritme
    - de gefitte preprocessing pipeline om te komen van ruwe features tot een inputset voor prediction
    """
    logger.info(f"Downloading model {version_name} (version {version_number}) from Azure Machine Learning")
    credential = DefaultAzureCredential()
    ml_client = MLClient(
        subscription_id=SUBSCRIPTIONID, resource_group_name=RGNAME, workspace_name=WORKSPACE_NAME, credential=credential
    )

    models = ml_client.models.list(name=azure.project_name)
    model_info = next(model for model in models if model.tags["version_name"] == version_name)

    name = model_info.tags["version_name"]
    version = model_info.version
    filename = os.path.basename(model_info.path)

    assert (version == version_number) & (name == version_name), "Model name and version don't match!"

    download_path = Path(f"{MODEL_DIR}/from_aml")
    os.makedirs(download_path, exist_ok=True)
    ml_client.models.download(name=azure.project_name, version=version, download_path=download_path)

    model_dict = load_from_pkl(os.path.join(download_path, azure.project_name, filename))

    return model_dict
