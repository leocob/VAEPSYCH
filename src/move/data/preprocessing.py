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

def scale(x: np.array, split_mask, names, interim_data_path, input_config_name) -> tuple[FloatArray, BoolArray]:
    """Center to mean and scale to unit variance. Convert NaN values to 0.

    Args:
        x: 2D array with samples in its rows and features in its columns

    Returns:
        Tuple containing (1) scaled output and (2) a 1D mask marking columns
        (i.e., features) without zero variance
    """
    x_train = x[split_mask]
    x_test = x[~split_mask]

    # Do I want a log transformation?
    logx_train = np.log2(x_train + 1)
    logx_test = np.log2(x_test + 1)


    mask_1d = ~np.isclose(np.nanstd(logx_train, axis=0), 0.0)
    
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()

    scaled_x_train = scaler.fit_transform(imputer.fit_transform(logx_train[:, mask_1d]))
    scaled_x_test = scaler.transform(imputer.transform(logx_test[:, mask_1d]))

    scaled_x = np.concatenate((scaled_x_train, scaled_x_test), axis=0)

    # print mean of means

    print(f"Mean of means of scaled_x_train {scaled_x_train.mean(axis=0).mean()}")
    print(f"Mean of stds of scaled_x_train {scaled_x_train.std(axis=0).mean()}")
    print(f"Mean of means of scaled_x_test {scaled_x_test.mean(axis=0).mean()}")
    print(f"Mean of stds of scaled_x_test {scaled_x_test.std(axis=0).mean()}")
    print(f"Mean of means of scaled_x {scaled_x.mean(axis=0).mean()}")
    print(f"Mean of stds of scaled_x {scaled_x.std(axis=0).mean()}")
 
    # plot_distr(x_train, logx_train, scaled_x_train, names, interim_data_path, input_config_name)
    
    return scaled_x, mask_1d



# def plot_distr(data_before_log, data_after_log, data_after_log_scaled, names, interim_data_path, input_config_name):

#     n_features = data_before_log.shape[1]
#     n_cols = 3
#     n_rows = 5  # Number of features to plot per page
#     n_pages = ceil(n_features / n_rows)

#     for page in range(n_pages):
#         fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
#         fig.suptitle(f'Distribution of features (Page {page+1}/{n_pages})', fontsize=16)

#         for i in range(n_rows):
#             feature_idx = page * n_rows + i
#             if feature_idx >= n_features:
#                 break

#             feature_name = names[feature_idx]

#             # Before log transformation
#             sns.histplot(data_before_log[:, feature_idx], kde=True, ax=axes[i, 0])
#             axes[i, 0].set_title(f'{feature_name}\nBefore Log')
#             axes[i, 0].set_xlabel('Value')

#             # After log transformation
#             sns.histplot(data_after_log[:, feature_idx], kde=True, ax=axes[i, 1])
#             axes[i, 1].set_title(f'{feature_name}\nAfter Log')
#             axes[i, 1].set_xlabel('Value')

#             # After scaling
#             sns.histplot(data_after_log_scaled[:, feature_idx], kde=True, ax=axes[i, 2])
#             axes[i, 2].set_title(f'{feature_name}\nAfter Scaling')
#             axes[i, 2].set_xlabel('Value')

#         # Remove any unused subplots
#         for i in range(feature_idx % n_rows + 1, n_rows):
#             for j in range(n_cols):
#                 fig.delaxes(axes[i, j])

#         plt.tight_layout()
#         plt.savefig(f'{interim_data_path}/{input_config_name}_page{page+1}.pdf')
#         plt.close()

#     print(f"Plots saved to {interim_data_path}/{input_config_name}_page*.pdf")

#     return None