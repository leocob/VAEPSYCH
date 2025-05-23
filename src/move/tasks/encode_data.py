__all__ = ["encode_data"]

from pathlib import Path

import numpy as np
import pandas as pd

from move.conf.schema import DataConfig
from move.core.logging import get_logger
from move.data import io, preprocessing
import warnings

def encode_data(config: DataConfig):
    """Encodes categorical and continuous datasets specified in configuration.
    Categorical data is one-hot encoded, whereas continuous data is z-score
    normalized.

    Args:
        config: data configuration
    """
    logger = get_logger(__name__)
    logger.info("Beginning task: encode data")

    p = config.percentage_threshold
    cases_names = Path(config.cases_names)
    raw_data_path = Path(config.raw_data_path)
    raw_data_path.mkdir(exist_ok=True)
    interim_data_path = Path(config.interim_data_path)
    interim_data_path.mkdir(exist_ok=True, parents=True)

    sample_names = io.read_names(raw_data_path / f"{config.sample_names}.txt")

    cases_names = pd.read_csv(raw_data_path / f"{config.cases_names}.tsv", sep = "\t")
    cases_names.query("SCZSPEC == 1", inplace = True)
    cases_names = cases_names["ID"].values

    
    # I need to create the split mask
    split_path = interim_data_path / "split_mask.npy"
    train_test_splits_file_name = Path(config.train_test_splits_file_name)
    train_test_path = raw_data_path / train_test_splits_file_name
    print(train_test_path)

    # if not train_test_path.exists():
    #     warnings.warn(f"Train test split file not found: {train_test_path}. Are you sure you don't want to use a train test split?")
    #     train_test_splits = None
    
    # else:
        # Will read train test split everytime I run the encoding_data, even if the split mask is already there
        # So I'm sure that the split mask is always the most recent one, as long as I don't move the train_test_splits.tsv file
    train_test_splits = pd.read_csv(train_test_path, sep = "\t")
    split_mask = train_test_splits["Split"].values == "train"
    np.save(split_path, split_mask)


    mappings = {}
    for dataset_name in config.categorical_names:
        logger.info(f"Encoding '{dataset_name}'")
        filepath = raw_data_path / f"{dataset_name}.tsv"
        # TODO: implement change p from command line and fix the handling of the NAs because it's wrong. It's removing a lot of features
        names, values, data = io.read_tsv(filepath, sample_names, cases_names, input_type = "categorical", p = p, interim_data_path = interim_data_path)
        values, mapping = preprocessing.one_hot_encode(values)
        mappings[dataset_name] = mapping
        io.dump_names(interim_data_path / f"{dataset_name}.txt", names)
        np.save(interim_data_path / f"{dataset_name}.npy", values)

        # create a pandas dataframe with the one hot encoded values
        # data = pd.DataFrame(values.reshape(value.shape[0],-1), columns=names)
        # print(data)
        # print(values)
        data.to_csv(interim_data_path / f"{dataset_name}.tsv", sep="\t", index=True)
    if mappings:
        io.dump_mappings(interim_data_path / "mappings.json", mappings)


    for input_config in config.continuous_inputs:
        # print(input_config)
        # print(str(input_config))
        scale = not hasattr(input_config, "scale") or input_config.scale
        action_name = "Encoding" if scale else "Reading"
        logger.info(f"{action_name} '{input_config.name}'")
        filepath = raw_data_path / f"{input_config.name}.tsv"
        names, values, data = io.read_tsv(filepath, sample_names, cases_names, input_type = "continuous", p = p, interim_data_path = interim_data_path)
        # print("continuous input")
        # print(f"names: {names}")
        # print(f"values: {values}")
        # print(f"values[4797]: {values[4797,:]}")
        if values is None:
            logger.warning(f"No data found for '{input_config.name}'")
            continue

        else:

            if scale:
                input_config_name = input_config.name
                # print(values)
                # uncomment for genomeDK
                values, mask_1d, scaled_x_df = preprocessing.scale(values, data, train_test_splits, split_mask, names, interim_data_path, input_config_name)
                # print(f"values: {values}")
                # print(f"values[4797]: {values[4797,:]}")
                # values, mask_1d = preprocessing.scale(values)

                cols_zero_variance = data.loc[:,~mask_1d].columns
                if len(cols_zero_variance) > 0:
                    print(f"Columns with zero variance: {cols_zero_variance.tolist()}")
                    # convert pandas index to list
                    names_cols_zero_variance = cols_zero_variance.tolist()
                else:
                    print("No columns with zero variance")
                    names_cols_zero_variance = []
                names = names[mask_1d]
                # print(f"Sum of Columns with zero variance: {np.sum(~mask_1d)}")
                logger.debug(f"Columns with zero variance: {np.sum(~mask_1d)}")
                
            io.dump_names(interim_data_path / f"{input_config.name}.txt", names)
            io.dump_names(interim_data_path / f"{input_config.name}_cols_zero_variance.txt", names_cols_zero_variance)
            # convert values to pandas dataframe

            # values_df = pd.DataFrame(values, columns=names)
            # values_df.to_csv(interim_data_path / f"{input_config.name}.tsv", sep="\t", index=True)
            scaled_x_df.to_csv(interim_data_path / f"{input_config.name}.tsv", sep="\t", index=True)
            np.save(interim_data_path / f"{input_config.name}.npy", values)



