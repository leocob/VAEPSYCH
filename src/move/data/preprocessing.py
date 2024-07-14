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


# def scale(x: np.ndarray) -> tuple[FloatArray, BoolArray]:
#     """Center to mean and scale to unit variance. Convert NaN values to 0.

#     Args:
#         x: 2D array with samples in its rows and features in its columns

#     Returns:
#         Tuple containing (1) scaled output and (2) a 1D mask marking columns
#         (i.e., features) without zero variance
#     """
#     logx = np.log2(x + 1)
#     mask_1d = ~np.isclose(np.nanstd(logx, axis=0), 0.0)
#     scaled_x = standardize(logx[:, mask_1d], axis=0)
#     scaled_x[np.isnan(scaled_x)] = 0
#     return scaled_x, mask_1d

# def scale(x: np.array, train_test_splits, split_mask, names, interim_data_path, input_config_name) -> tuple[FloatArray, BoolArray]:
#     """Center to mean and scale to unit variance. Convert NaN values to 0.

#     Args:
#         x: 2D array with samples in its rows and features in its columns

#     Returns:
#         Tuple containing (1) scaled output and (2) a 1D mask marking columns
#         (i.e., features) without zero variance
#     """

#     imputer = SimpleImputer(strategy='mean')
#     scaler = StandardScaler()
#     if train_test_splits is None:

        
#         mask_1d = ~np.isclose(np.nanstd(x, axis=0), 0.0)

#         scaled_x = scaler.fit_transform((x[:, mask_1d]))

#         scaled_x[np.isnan(scaled_x)] = 0  
#         # scaled_x = standardize(x[:, mask_1d], axis=0)


#         # print(f"Mean of means of scaled_x {scaled_x.mean(axis=0).mean()}")
#         # print(f"Mean of stds of scaled_x {scaled_x.std(axis=0).mean()}")

#         plot_distr(x, scaled_x, names, interim_data_path, input_config_name)


#     else:

#         print(f"Split mask: {split_mask}")
#         print(f"Train test splits: {train_test_splits}")
#         x_train = x[split_mask]
#         x_test = x[~split_mask]

#         mask_1d = ~np.isclose(np.nanstd(x_train, axis=0), 0.0)
#         # I make sure the mask is True for all of them, meaning that all the columns have std != 0
#         # print(f"mask_1d: {mask_1d}")
#         # print(f"mask1d shape: {mask_1d.shape}")
        
#         # print(f"x_test[5997,:]: {x_test[5997,:]}")

#         # print(f"x_train[:, mask_1d]: {x_train[:, mask_1d]}")
#         x_traindf = pd.DataFrame(x_train, columns=names)
#         print(f"x_traindf:\n {x_traindf}")
#         scaled_x_train = scaler.fit_transform(x_train[:, mask_1d])
#         scaled_xtraindf = pd.DataFrame(scaled_x_train, columns=names[mask_1d])
#         print(f"scaled_xtraindf:\n {scaled_xtraindf}")
#         # scaled_x_train[np.isnan(scaled_x_train)] = 0
#         # print(f"scaled_x_train: {scaled_x_train}")
#         scaled_x_test = scaler.transform(x_test[:, mask_1d])
#         # scaled_x_test[np.isnan(scaled_x_test)] = 0
#         # print(f"scaled_x_test[5997,:]: {scaled_x_test[5997,:]}")

#         scaled_x = np.concatenate((scaled_x_train, scaled_x_test), axis=0)

#         # print index of scaled_x
#         # convert scaled_x to pandas dataframe
#         scaled_x_df = pd.DataFrame(scaled_x, columns=names[mask_1d])
#         # print scaled_x_df from index 0 to 4799
#         print(f"scaled_x_df: \n{scaled_x_df.iloc[:4800]}")
#         print(f"scaled_x_df: \n{scaled_x_df.query('index == 4797')}")
#         # print(f"scaled_x_df[5997,:]: {scaled_x_df.iloc[5997,:]}")
#         # print(f"scaled_x: {scaled_x}")
#         # print(f"scaled_x[5997,:]: {scaled_x[5997,:]}")
#         # print mean of means

#         # print(f"Mean of means of scaled_x_train {scaled_x_train.mean(axis=0).mean()}")
#         # print(f"Mean of stds of scaled_x_train {scaled_x_train.std(axis=0).mean()}")
#         # print(f"Mean of means of scaled_x_test {scaled_x_test.mean(axis=0).mean()}")
#         # print(f"Mean of stds of scaled_x_test {scaled_x_test.std(axis=0).mean()}")
#         # print(f"Mean of means of scaled_x {scaled_x.mean(axis=0).mean()}")
#         # print(f"Mean of stds of scaled_x {scaled_x.std(axis=0).mean()}")
    
#         # plot_distr(x_train, x_train, scaled_x_train, names, interim_data_path, input_config_name)

#     return scaled_x, mask_1d



# scale on dataframe and not on numpy array. I want to see the damn IDs
def scale(x: np.array, data, train_test_splits, split_mask, names, interim_data_path, input_config_name) -> tuple[FloatArray, BoolArray]:
    """Center to mean and scale to unit variance. Convert NaN values to 0.

    Args:
        x: 2D array with samples in its rows and features in its columns

    Returns:
        Tuple containing (1) scaled output and (2) a 1D mask marking columns
        (i.e., features) without zero variance
    """

    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    x = data

    print(f"x \n{x}")
    if train_test_splits is None:

        
        mask_1d = ~np.isclose(np.nanstd(x, axis=0), 0.0)

        scaled_x = scaler.fit_transform((x[:, mask_1d]))

        scaled_x[np.isnan(scaled_x)] = 0  
        # scaled_x = standardize(x[:, mask_1d], axis=0)


        # print(f"Mean of means of scaled_x {scaled_x.mean(axis=0).mean()}")
        # print(f"Mean of stds of scaled_x {scaled_x.std(axis=0).mean()}")

        plot_distr(x, scaled_x, names, interim_data_path, input_config_name)


    else:

        print(f"Split mask: {split_mask}")
        print(f"Train test splits: {train_test_splits}")
        x_train = x[split_mask]
        x_test = x[~split_mask]

        mask_1d = ~np.isclose(np.nanstd(x_train, axis=0), 0.0)
        # I make sure the mask is True for all of them, meaning that all the columns have std != 0
        # print(f"mask_1d: {mask_1d}")
        # print(f"mask1d shape: {mask_1d.shape}")
        
        # print(f"x_test[5997,:]: {x_test[5997,:]}")

        # print(f"x_train[:, mask_1d]: {x_train[:, mask_1d]}")
        # x_traindf = pd.DataFrame(x_train, columns=names)
        print(f"x_train:\n {x_train}")
        print(f"x_train[:, mask_1d]: {x_train.loc[:, mask_1d]}")
        scaled_x_train = scaler.fit_transform(x_train.loc[:, mask_1d])
        # scaled_x_train = pd.DataFrame(scaled_x_train, columns=data[mask_1d].columns)
        # scaled_xtraindf = pd.DataFrame(scaled_x_train, columns=names[mask_1d])
        print(f"scaled_x_train:\n {scaled_x_train}")
        scaled_x_train[np.isnan(scaled_x_train)] = 0
        # print(f"scaled_x_train: {scaled_x_train}")
        scaled_x_test = scaler.transform(x_test.loc[:, mask_1d])
        # scaled_x_test = pd.DataFrame(scaled_x_test, columns=data[mask_1d].columns)
        scaled_x_test[np.isnan(scaled_x_test)] = 0
        # print(f"scaled_x_test[5997,:]: {scaled_x_test[5997,:]}")

        print(f"scaled_x_test:\n {scaled_x_test}")
        # print type
        print(f"type of scaled_x_train: {type(scaled_x_train)}")

        scaled_x = np.concatenate((scaled_x_train, scaled_x_test), axis=0)
        print(f"scaled_x:\n {scaled_x}")
        print(f"scaled_x[4797,:]: {scaled_x[4797,:]}")


        scaled_x_df = pd.DataFrame(scaled_x, columns=x.loc[:,mask_1d].columns)

        # print index of scaled_x
        # convert scaled_x to pandas dataframe
        # scaled_x_df = pd.DataFrame(scaled_x, columns=names[mask_1d])
        # print scaled_x_df from index 0 to 4799
        print(f"scaled_x_df: \n{scaled_x_df}")
        print(f"scaled_x_df.iloc[:4800]: \n{scaled_x_df.iloc[:4800]}")
        print(f"scaled_x_df: \n{scaled_x_df.query('index == 4797')}")
        # print(f"scaled_x_df[5997,:]: {scaled_x_df.iloc[5997,:]}")
        # print(f"scaled_x: {scaled_x}")
        # print(f"scaled_x[5997,:]: {scaled_x[5997,:]}")
        # print mean of means

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