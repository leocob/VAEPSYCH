__all__ = [
    "dump_names",
    "dump_mappings",
    "load_mappings",
    "load_preprocessed_data",
    "read_config",
    "read_names",
    "read_tsv",
]

import json
from pathlib import Path
from typing import Optional
from os.path import basename
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from move import HYDRA_VERSION_BASE, conf
from move.conf.schema import AnalyzeLatentConfig, MOVEConfig
from move.core.typing import BoolArray, FloatArray, ObjectArray, PathLike
from move.core.logging import get_logger

def read_config(
    data_config_name: Optional[str], task_config_name: Optional[str], *args
) -> DictConfig:
    """Composes configuration for the MOVE framework.

    Args:
        data_config_name: Name of data configuration file
        task_config_name: Name of task configuration file
        *args: Additional overrides

    Returns:
        Merged configuration
    """
    overrides = []
    if data_config_name is not None:
        overrides.append(f"data={data_config_name}")
    if task_config_name is not None:
        overrides.append(f"task={task_config_name}")
    overrides.extend(args)
    with hydra.initialize_config_module(conf.__name__, version_base=HYDRA_VERSION_BASE):
        return hydra.compose("main", overrides=overrides)


def load_categorical_dataset(filepath: PathLike) -> FloatArray:
    """Loads categorical data in a NumPy file.

    Args:
        filepath: Path to NumPy file containing a categorical dataset

    Returns:
        NumPy array containing categorical data
    """
    return np.load(filepath).astype(np.float32)


def load_continuous_dataset(filepath: PathLike) -> tuple[FloatArray, BoolArray]:
    """Loads continuous data from a NumPy file and filters out columns
    (features) whose sum is zero. Additionally, encodes NaN values as zeros.

    Args:
        filepath: Path to NumPy file containing a continuous dataset

    Returns:
        Tuple containing (1) the NumPy dataset and (2) a mask marking columns
        (i.e., features) that were not filtered out
    """
    data = np.load(filepath).astype(np.float32)
    data[np.isnan(data)] = 0
    mask_col = np.abs(data).sum(axis=0) != 0
    data = data[:, mask_col]
    return data, mask_col


def load_preprocessed_data(
    path: Path,
    categorical_dataset_names: list[str],
    continuous_dataset_names: list[str],
) -> tuple[list[FloatArray], list[list[str]], list[FloatArray], list[list[str]]]:
    """Loads the pre-processed categorical and continuous data.

    Args:
        path: Where the data is saved
        categorical_dataset_names: List of names of the categorical datasets
        continuous_dataset_names: List of names of the continuous datasets

    Returns:
        Returns two pairs of list containing (1, 3) the pre-processed data and
        (2, 4) the lists of names of each feature
    """

    categorical_data, categorical_var_names = [], []
    for dataset_name in categorical_dataset_names:
        data = load_categorical_dataset(path / f"{dataset_name}.npy")
        categorical_data.append(data)
        var_names = read_names(path / f"{dataset_name}.txt")
        categorical_var_names.append(var_names)

    continuous_data, continuous_var_names = [], []
    for dataset_name in continuous_dataset_names:
        data, keep = load_continuous_dataset(path / f"{dataset_name}.npy")
        continuous_data.append(data)
        var_names = read_names(path / f"{dataset_name}.txt")
        var_names_not_kept = [name for i, name in enumerate(var_names) if not keep[i]]
        var_names = [name for i, name in enumerate(var_names) if keep[i]]

        # print(f"dataset_name: {dataset_name}")
        # print(f"data.shape: {data.shape}")
        # # print(f"data: {data}")
        # print(f"var_names: {var_names}")
        # print(f"var_names_not_kept: {var_names_not_kept}")
        continuous_var_names.append(var_names)

    return (
        categorical_data,
        categorical_var_names,
        continuous_data,
        continuous_var_names,
    )


def read_names(path: PathLike) -> list[str]:
    """Reads sample names from a text file. The text file should have one line
    per sample name.

    Args:
        path: Path to the text file

    Returns:
        A list of sample names
    """
    with open(path, "r", encoding="utf-8") as file:
        return [i.strip() for i in file.readlines()]


def read_tsv(
    path: PathLike, sample_names: Optional[list[str]] = None, cases_names: Optional[list[str]] = None, input_type: str = "categorical", p: float = 0, interim_data_path: Path = None
) -> tuple[ObjectArray, np.ndarray]:
    """Read a dataset from a TSV file. The TSV is expected to have an index
    column (0th index).

    Args:
        path: Path to TSV
        index: List of sample names used to sort/filter samples

    Returns:
        Tuple containing (1) feature names and (2) 2D matrix (samples x
        features)
    """
    
    logger = get_logger(__name__)
    data = pd.read_csv(path, index_col=0, sep="\t")

    data_cases = data.copy()
    # get the name of the dataset
    dataset_name = basename(path)
    # remove extension
    dataset_name = dataset_name.split(".")[0]

    # interim_path = Path(config.data.interim_data_path)
    # print("Before filtering in read_tsv function")
    # print(data)
    # print(f"data.query(index == 610): {data.query('index == 610')}")
    # print(f"data.query(ID==1219925): {data.query('ID==1219925')}")

    if sample_names and cases_names is not None:
        # print(f"length Sample names: {len(sample_names)}") # 6000
        data.index = data.index.astype(str, False) # 22092
        data = data.loc[sample_names]

        data_cases = data_cases.loc[cases_names]

        # print(f"data after filtering for sample name:\n{data}") # 6000 with IDS
        # # print("After selecting for my samples")
        # print(data)
        # print(f"data.query(index == 610): {data.query('index == 610')}")
        # print(f"data.query(index == 172): {data.query('index == 172')}")
        # print(f"data.query(ID==1219925): {data.query('ID==1219925')}")

        
    if input_type == "categorical":

        ones_stats = pd.DataFrame({
            "Percentage": (data_cases == 1).mean(),
            "Count": (data_cases == 1).sum(),
            "Threshold": p
        })

        columns_to_keep = ones_stats[ones_stats["Percentage"] >= p].index

        columns_removed = ones_stats[ones_stats["Percentage"] < p].index

        columns_removed = columns_removed.tolist()



        with open(interim_data_path / f"{dataset_name}_features_kept_more_than_{p}.txt", "w") as file:
            for column in columns_to_keep:
                file.write(f"{column}\n")


        with open(interim_data_path / f"{dataset_name}_features_removed_less_than_{p}.txt", "w") as file:
            for column in columns_removed:
                file.write(f"{column}\n")

        ones_stats.to_csv(interim_data_path / f"{dataset_name}_stats.tsv", sep="\t", index=True)

        if columns_to_keep.empty:
            logger.warning(f"No columns with more than {p} non-NAs in dataset {dataset_name}")
            return None, None, None

        else:
            columns_to_keep = columns_to_keep.tolist()
            data = data[columns_to_keep]

    elif input_type == "continuous":

        nonas_stats = pd.DataFrame({
            "Percentage": data_cases.notna().mean(),
            "Count": data_cases.notna().sum(),
            "Threshold": p
        })

        columns_to_keep = nonas_stats[nonas_stats["Percentage"] >= p].index
        columns_removed = nonas_stats[nonas_stats["Percentage"] < p].index


        
        columns_removed = columns_removed.tolist()

        with open(interim_data_path / f"{dataset_name}_features_removed_less_than_{p}.txt", "w") as file:
            for column in columns_removed:
                file.write(f"{column}\n")

        with open(interim_data_path / f"{dataset_name}_features_kept_more_than_{p}.txt", "w") as file:
            for column in columns_to_keep:
                file.write(f"{column}\n")

        nonas_stats.to_csv(interim_data_path / f"{dataset_name}_stats.tsv", sep="\t", index=True)

        if columns_to_keep.empty:
            logger.warning(f"No columns with more than {p} non-NAs in dataset {dataset_name}")
            return None, None, None

        else:
            columns_to_keep = columns_to_keep.tolist()
            data = data[columns_to_keep]
                
        
    # TODO: add sweetviz report

        # print(f"data.index.values")
        # print(data.index.values)
    return data.columns.values, data.values, data 


def load_mappings(path: PathLike) -> dict[str, dict[str, int]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def dump_mappings(path: PathLike, mappings: dict[str, dict[str, int]]) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(mappings, file, indent=4, ensure_ascii=False)


def dump_names(path: PathLike, names: np.ndarray) -> None:
    with open(path, "w", encoding="utf-8") as file:
        file.writelines([f"{name}\n" for name in names])
