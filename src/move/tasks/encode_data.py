__all__ = ["encode_data"]

from pathlib import Path

import numpy as np
import pandas as pd

from move.conf.schema import DataConfig
from move.core.logging import get_logger
from move.data import io, preprocessing


def encode_data(config: DataConfig):
    """Encodes categorical and continuous datasets specified in configuration.
    Categorical data is one-hot encoded, whereas continuous data is z-score
    normalized.

    Args:
        config: data configuration
    """
    logger = get_logger(__name__)
    logger.info("Beginning task: encode data")

    raw_data_path = Path(config.raw_data_path)
    raw_data_path.mkdir(exist_ok=True)
    interim_data_path = Path(config.interim_data_path)
    interim_data_path.mkdir(exist_ok=True, parents=True)

    sample_names = io.read_names(raw_data_path / f"{config.sample_names}.txt")


    train_test_path = raw_data_path / "train_test_splits.tsv"

    if not train_test_path.exists():
        raise FileNotFoundError(f"Train test split file not found: {train_test_path}")

    # I need to create the split mask
    split_path = interim_data_path / "split_mask.npy"

    # Will read train test split everytime I run the encoding_data, even if the split mask is already there
    # So I'm sure that the split mask is always the most recent one, as long as I don't move the train_test_splits.tsv file

    train_test_splits = pd.read_csv(train_test_path, sep = "\t")
    split_mask = train_test_splits["Split"].values == "train"
    np.save(split_path, split_mask)


    mappings = {}
    for dataset_name in config.categorical_names:
        logger.info(f"Encoding '{dataset_name}'")
        filepath = raw_data_path / f"{dataset_name}.tsv"
        names, values = io.read_tsv(filepath, sample_names, input_type = "categorical", p = 0.01)

        values, mapping = preprocessing.one_hot_encode(values)
        mappings[dataset_name] = mapping
        io.dump_names(interim_data_path / f"{dataset_name}.txt", names)
        np.save(interim_data_path / f"{dataset_name}.npy", values)
    if mappings:
        io.dump_mappings(interim_data_path / "mappings.json", mappings)

    for input_config in config.continuous_inputs:
        scale = not hasattr(input_config, "scale") or input_config.scale
        action_name = "Encoding" if scale else "Reading"
        logger.info(f"{action_name} '{input_config.name}'")
        filepath = raw_data_path / f"{input_config.name}.tsv"
        names, values = io.read_tsv(filepath, sample_names, input_type = "continuous", p = 0.01)


        if scale:
            input_config_name = input_config.name
            values, mask_1d = preprocessing.scale(values, split_mask, names, interim_data_path, input_config_name)
            # values, mask_1d = preprocessing.scale(values, split_mask)
            names = names[mask_1d]
            logger.debug(f"Columns with zero variance: {np.sum(~mask_1d)}")
        io.dump_names(interim_data_path / f"{input_config.name}.txt", names)
        np.save(interim_data_path / f"{input_config.name}.npy", values)



