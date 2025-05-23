__all__ = ["one_hot_encode", "one_hot_encode_single", "scale"]

from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.preprocessing import scale as standardize
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from math import ceil
from matplotlib.backends.backend_pdf import PdfPages


from move.core.typing import BoolArray, FloatArray, IntArray


def _category_name(value: Any) -> str:
    return value if isinstance(value, str) else str(int(value))


def one_hot_encode(x_: ArrayLike) -> tuple[IntArray, dict[str, int]]:
    """One-hot encode a matrix with samples in its rows and features in its
    columns. Columns share number of classes.

    Args:
        x: a 1D or 2D matrix, can be numerical or contain strings

    Returns:
        A 3D one-hot encoded matrix (extra dim corresponds to number of
        classes) and a mapping between classes and corresponding codes
    """
    x: np.ndarray = np.copy(x_)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    shape = x.shape
    has_na = np.any(pd.isna(x))
    if x.dtype == object:
        x = x.astype(str)
    categories, codes = np.unique(x, return_inverse=True)
    num_classes = len(categories)
    encoded_x = np.zeros((x.size, num_classes), dtype=np.uint8)
    encoded_x[np.arange(x.size), codes.astype(np.uint8).ravel()] = 1
    encoded_x = encoded_x.reshape(*shape, num_classes)
    if has_na:
        # remove NaN column
        categories = categories[:-1]
        encoded_x = encoded_x[:, :, :-1]
    mapping = {
        _category_name(category): code for code, category in enumerate(categories)
    }
    return encoded_x, mapping


def one_hot_encode_single(mapping: dict[str, int], value: Optional[str]) -> IntArray:
    """One-hot encode a single value given an existing mapping.

    Args:
        mapping: cateogry-to-code lookup dictionary
        value: category

    Returns:
        2D array
    """
    encoded_value = np.zeros((1, len(mapping)))
    if not pd.isna(value):
        code = mapping[str(value)]
        encoded_value[0, code] = 1
    return encoded_value


def scale(x: np.array, data, train_test_splits, split_mask, names, interim_data_path, input_config_name) -> tuple[FloatArray, BoolArray]:
    """Center to mean and scale to unit variance. Convert NaN values to 0.
    Perform operations on pandas dataframes preserving the IDs and the original order

    Args:
        x: 2D array with samples in its rows and features in its columns

    Returns:
        Tuple containing (1) scaled output and (2) a 1D mask marking columns
        (i.e., features) without zero variance
    """

    # print every input


    if input_config_name == "hosp_contacts":
        data = np.log(data+1)

    if input_config_name == "birth_related_cont_features":
        # log transform column "n_prev_completed_pregnancies"
        data["n_prev_completed_pregnancies"] = np.log(data["n_prev_completed_pregnancies"]+1)


    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    x = data




    # TODO: REMEMBER TO UNCOMMENT THIS FOR GENOMEDK
    # x = np.log2(x+1)
    # print(f"x: {x}")
    # print(f"x.shape: {x.shape}")
    # print(f"x.query(ID==1219925):")
    # row = x[x.index == "1219925"]
    # print(f"row: {row}")
    # row_number = x.index.get_indexer(row.index)[0]
    # print(f"row_number: {row_number}")

    # x is a pandas dataframe. Keep only the rows that are in the train set
    # x = data  # Assuming this isasdasdads your original dataframe

    # Create a copy of x to preserve the original order
    x_copy = x.copy()

    # get order of x_copy index
    # orig_index_order = x_copy.index
    # print(f"len(orig_index_order):{len(orig_index_order)}")
    # print(f"orig_index_order: {orig_index_order}")

    # Split the data
    x_train = x.iloc[split_mask]
    x_test = x.iloc[~split_mask]

    mask_1d = ~np.isclose(np.nanstd(x_train, axis=0), 0.0)
    # print(f"x.shape: {x.shape}")
    # print(f"mask_1d.shape: {mask_1d.shape}")

    # keep only the columns with std !=0

    # Scale the training data
    scaler.fit(x_train.loc[:, mask_1d])

    # Scale both training and test data
    scaled_x_train = pd.DataFrame(
        scaler.transform(x_train.loc[:, mask_1d]),
        columns=x_train.loc[:, mask_1d].columns,
        index=x_train.index
    )

    scaled_x_test = pd.DataFrame(
        scaler.transform(x_test.loc[:, mask_1d]),
        columns=x_test.loc[:, mask_1d].columns,
        index=x_test.index
    )

    # print(f"Printing number of zeros per column BEFORE NA -> 0 for {input_config_name}")
    # # print their sum, so how many values are 0
    # print(f"Sum of zeros in scaled_x_train: {np.sum(scaled_x_train == 0, axis=0)}")
    # print(f"Sum of zeros in scaled_x_test: {np.sum(scaled_x_test == 0, axis=0)}")

    n_0_train = np.sum(scaled_x_train.values == 0)
    n_0_test = np.sum(scaled_x_test.values == 0)

    n_0_train_test = n_0_train + n_0_test
    # print(f"BEFORE NA -> 0 Sum of zeros in scaled_x_train + scaled_x_test: {n_0_train_test}")
    if n_0_train_test > 0:
        raise ValueError("There are zeros in the scaled data before setting NA to 0. This should not happen.")
    # Replace NaN values with 0
    scaled_x_train[np.isnan(scaled_x_train)] = 0
    scaled_x_test[np.isnan(scaled_x_test)] = 0

    # print(f"Printing number of zeros per column AFTER NA -> 0 for {input_config_name}")
    # # print their sum, so how many values are 0
    # print(f"Sum of zeros in scaled_x_train: {np.sum(scaled_x_train == 0, axis=0)}")
    # print(f"Sum of zeros in scaled_x_test: {np.sum(scaled_x_test == 0, axis=0)}")

    n_0_train = np.sum(scaled_x_train.values == 0)
    n_0_test = np.sum(scaled_x_test.values == 0)

    n_0_train_test = n_0_train + n_0_test
    # print(f"AFTER NA -> 0 Sum of zeros in scaled_x_train + scaled_x_test: {n_0_train_test}")




    # Create a DataFrame with the same index as the original x
    scaled_x_df = pd.DataFrame(index=x_copy.index)


    # print(f"scaled_x_train: {scaled_x_train}")
    # print(f"scaled_x_test: {scaled_x_test}")

    # 
    # Fill in the scaled values
    scaled_x_df.loc[scaled_x_train.index, scaled_x_train.columns] = scaled_x_train
    scaled_x_df.loc[scaled_x_test.index, scaled_x_test.columns] = scaled_x_test


    # keep only the columns with std !=0
    # scaled_x_df = scaled_x_df.loc[:, mask_1d]

    # print(f"scaled_x_df.shape: {scaled_x_df.shape}")
    # print(f"scaled_x_df: {scaled_x_df}")

    # Add back any columns that weren't scaled (where mask_1d is False)
    # for col in x_copy.columns[~mask_1d]:
    #     scaled_x_df[col] = x[col]

    # # Ensure the column order matches the original x
    # scaled_x_df = scaled_x_df[x_copy.columns]

    # convert to numpy array
    scaled_x = scaled_x_df.to_numpy()

    # print("scaled_x_df[scaled_x_df.index == 1219925]")
    # row = scaled_x_df[scaled_x_df.index == "1219925"]
    # print(f"row: {row}")
    # row_number = scaled_x_df.index.get_indexer(row.index)[0]
    # print(f"row_number: {row_number}")

    # print(f"x.shape: {x.shape}")
    # print(f"scaled_x_df.shape: {scaled_x_df.shape}")


    # x_num = x.to_numpy()
    # print(f"x_num.shape: {x_num.shape}")
    # print(f"x_num[170,:]")
    # print(x_num[170,:])
    # print(f"x_num[139,:]")
    # print(x_num[139,:])

    # print("scaled_x[139,:]")
    # print(scaled_x[139,:])



    # print(f"Mean of means of scaled_x_train {scaled_x_train.mean(axis=0).mean()}")
    # print(f"Mean of stds of scaled_x_train {scaled_x_train.std(axis=0).mean()}")
    # print(f"Mean of means of scaled_x_test {scaled_x_test.mean(axis=0).mean()}")
    # print(f"Mean of stds of scaled_x_test {scaled_x_test.std(axis=0).mean()}")
    # print(f"Mean of means of scaled_x {scaled_x.mean(axis=0).mean()}")
    # print(f"Mean of stds of scaled_x {scaled_x.std(axis=0).mean()}")

    # plot_distr(x_train, x_train, scaled_x_train, names, interim_data_path, input_config_name)

    return scaled_x, mask_1d, scaled_x_df

def plot_distr(data_before_log, data_after_log, data_after_log_scaled, names, interim_data_path, input_config_name):
    n_features = data_before_log.shape[1]
    n_cols = 3
    n_rows = 10  # Number of features to plot per page
    n_pages = ceil(n_features / n_rows)

    pdf_path = f'{interim_data_path}/{input_config_name}.pdf'

    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))

            for i in range(n_rows):
                feature_idx = page * n_rows + i
                if feature_idx >= n_features:
                    break

                feature_name = names[feature_idx]

                # print(f"Type of data_before {type(data_before_log)}")
                # print(data_before_log)
                # print(data_after_log[:, feature_idx])
                # print(f"Type of data_after {type(data_after_log)}")
                # print(data_after_log)
                # print(f"Type of data_after_scaled {type(data_after_log_scaled)}")
                # print(data_after_log_scaled)

                # Before log transformation
                sns.histplot(data_before_log[:, feature_idx], kde=True, ax=axes[i, 0])
                axes[i, 0].set_title(f'{feature_name}\nTraining set distr before Log-transf')
                axes[i, 0].set_xlabel('Value')
                mean_before = np.nanmean(data_before_log[:, feature_idx], axis=0)
                std_before = np.nanstd(data_before_log[:, feature_idx], axis=0)
                axes[i, 0].text(0.95, 0.95, f'Mean: {mean_before:.2f}\nStd: {std_before:.2f}', 
                                verticalalignment='top', horizontalalignment='right', 
                                transform=axes[i, 0].transAxes, bbox=dict(facecolor='white', alpha=0.5))

                # After log transformation
                sns.histplot(data_after_log[:, feature_idx], kde=True, ax=axes[i, 1])
                axes[i, 1].set_title(f'{feature_name}\nTraining set distr after Log-transf')
                axes[i, 1].set_xlabel('Value')
                mean_after = np.nanmean(data_after_log[:, feature_idx], axis=0)
                std_after = np.nanstd(data_after_log[:, feature_idx], axis=0)
                axes[i, 1].text(0.95, 0.95, f'Mean: {mean_after:.2f}\nStd: {std_after:.2f}', 
                                verticalalignment='top', horizontalalignment='right', 
                                transform=axes[i, 1].transAxes, bbox=dict(facecolor='white', alpha=0.5))

                # After scaling
                sns.histplot(data_after_log_scaled[:, feature_idx], kde=True, ax=axes[i, 2])
                axes[i, 2].set_title(f'{feature_name}\nTraining set distr after Log + z-score norm')
                axes[i, 2].set_xlabel('Value')
                mean_scaled = np.nanmean(data_after_log_scaled[:, feature_idx], axis=0)
                std_scaled = np.nanstd(data_after_log_scaled[:, feature_idx], axis=0)
                axes[i, 2].text(0.95, 0.95, f'Mean: {mean_scaled:.2f}\nStd: {std_scaled:.2f}', 
                                verticalalignment='top', horizontalalignment='right', 
                                transform=axes[i, 2].transAxes, bbox=dict(facecolor='white', alpha=0.5))

            # Remove any unused subplots
            for j in range(i + 1, n_rows):
                for k in range(n_cols):
                    fig.delaxes(axes[j, k])

            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    print(f"Plots saved to {pdf_path}")

    return None