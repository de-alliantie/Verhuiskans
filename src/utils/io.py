import logging
import os
import pickle
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.settings import DATA_DIR


class LEVEL(Enum):
    """Enumeration representing different levels of data processing."""

    LOAD = 0
    PREPARE = 2
    TRAIN = 3
    PREDICT = 4


def load_df_from_csv(level: LEVEL, file_name: str, **kwargs) -> pd.DataFrame:
    """Load a DataFrame from a CSV file.

    Parameters:
        level (LEVEL): The level of the data directory.
        file_name (str): The name of the CSV file (without the extension).
        **kwargs: Additional keyword arguments to be passed to pd.read_csv() function.

    Returns:
        pd.DataFrame: A DataFrame containing the data from the CSV file.
    """
    p = generate_data_dir_path(level, file_name, suffix=".csv")
    return pd.read_csv(p, sep=";", index_col=0, **kwargs)


def save_df_to_csv(level: LEVEL, file_name: str, df: pd.DataFrame, **kwargs) -> Path:
    """Save a DataFrame to a CSV file.

    Parameters:
        level (LEVEL): The level of the data directory.
        file_name (str): The name of the CSV file (without the extension).
        df (pd.DataFrame): The DataFrame to be saved to the CSV file.
        **kwargs: Additional keyword arguments to be passed to pd.to_csv() function.

    Returns:
        Path: The path to the saved CSV file.
    """
    p = generate_data_dir_path(level, file_name, suffix=".csv")
    p.parent.mkdir(exist_ok=True, parents=True)
    logging.info(f"saving df to csv: {p}")
    df.to_csv(p, sep=";", **kwargs)
    return p


def generate_data_dir_path(level: LEVEL, file_name: str, suffix: str = "") -> Path:
    """Generate a path for a data directory.

    Parameters:
        level (LEVEL): The level of the data directory.
        file_name (str): The name of the file or directory.
        suffix (str, optional): The suffix to be appended to the file name. Defaults to "".

    Returns:
        Path: The path to the data directory.
    """
    p = Path(DATA_DIR) / level.name.lower() / file_name
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix == "":
        p = p.with_suffix(suffix)
    return p


def save_to_pkl(var, filepath: str, create_path: bool = True, **kwargs: dict) -> None:
    """Save variable to a path as pickle file. Kwargs are passed to save functions.

    Args:
        var: Variable to be written to file
        filepath (str): Path to write to
        create_path (bool, optional): Whether the path should be created. Defaults to False.

    Raises:
        FileNotFoundError: If path is not to a file
        ValueError: if filepath is not to .csv or .pickle
    """
    logging.info(f"saving var to pkl file : {filepath}")
    if not Path(filepath).is_file:
        raise FileNotFoundError("filepath should point to a file")
    if create_path:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    _save_to_pkl(var, filepath, **kwargs)


def _save_to_pkl(var, filepath: str | Path, **kwargs: int) -> None:
    allowed_suffixes = [".pickle"]
    suffix = Path(filepath).suffix
    if suffix == ".pickle":
        with open(filepath, "wb") as fp:
            pickle.dump(var, fp, **kwargs)
    else:
        raise ValueError(f'suffix {suffix} is not supported (only {", ".join(allowed_suffixes)})')


def load_from_pkl(filepath: str | Path, **kwargs: Optional[dict]) -> object:
    """Load variable from pickle file.

    Args:
        filepath (str): path to the pickle file to be loaded


    Raises:
        FileNotFoundError: If path is not to a file
        FileNotFoundError: If path does not exist

    Returns:
        var (object): variable that was loaded from file
    """
    logging.info(f"loading var from file : {filepath}")
    if not Path(filepath).is_file:
        raise FileNotFoundError("filepath should point to a file")
    if not Path(filepath).exists():
        raise FileNotFoundError("filepath does not exist")

    var = _load_from_pkl(filepath, **kwargs)
    return var


def _load_from_pkl(filepath, **kwargs) -> Any | None:
    allowed_suffixes = [".pickle"]
    suffix = Path(filepath).suffix
    if suffix == ".pickle":
        with open(filepath, "rb") as fp:
            var = pickle.load(fp, **kwargs)
    else:
        raise ValueError(f'suffix {suffix} is not supported (only {", ".join(allowed_suffixes)})')
    return var


def _load_model_locally(model_path: str) -> object:
    """Load a pickled model from local disk.

    Args:
        model_path (str): Path to a local pickled lifelines.CoxPHFitter model

    Returns:
        _type_: CoxPHFitter
    """
    print(f"Loading local model: {model_path}")

    with open(f"{model_path}", "rb") as f:
        coxph_model = pickle.load(f)

    return coxph_model


def _save_model_locally(model, model_filename: str) -> None:
    """Saves a model to local models folder.

    Args:
        model (CoxPHFitter): A lifelines.CoxPHFitter model
        model_filename (str): Desired filename

    Returns:
        _type_: None
    """
    model_save_path = f"models/{model_filename}"

    # Save model locally after training
    if not os.path.exists("models"):
        os.makedirs("models")

    print(f"Saving model to {model_save_path}")
    with open(model_save_path, "wb") as f:
        pickle.dump(model, f)
