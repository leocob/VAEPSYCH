__all__ = ["analyze_latent"]

import re
from pathlib import Path
from typing import Sized, cast

import hydra
import numpy as np
import pandas as pd
import torch
from sklearn.base import TransformerMixin

import move.visualization as viz
from move.analysis.metrics import (
    calculate_accuracy,
    calculate_cosine_similarity,
)
from move.conf.schema import AnalyzeLatentConfig, MOVEConfig
from move.core.logging import get_logger
from move.core.typing import FloatArray
from move.data import io
from move.data.dataloaders import MOVEDataset, make_dataloader
from move.data.perturbations import (
    perturb_categorical_data,
    perturb_continuous_data,
)
from move.data.preprocessing import one_hot_encode_single
from move.models.vae import VAE
from move.training.training_loop import TrainingLoopOutput


def find_feature_values(
    feature_name: str,
    feature_names_lists: list[list[str]],
    feature_values: list[FloatArray],
) -> tuple[int, FloatArray]:
    """Look for the feature in the list of datasets and returns its values.

    Args:
        feature_name: Look-up key
        feature_names_lists: List of lists with feature names for each dataset
        feature_values: List of data arrays, each representing a dataset

    Raises:
        KeyError: If feature does not exist in any dataset

    Returns:
        Tuple containing (1) index of dataset containing feature and (2)
        values corresponding to the feature
    """
    dataset_index, feature_index = [None] * 2
    for dataset_index, feature_names in enumerate(feature_names_lists):
        try:
            feature_index = feature_names.index(feature_name)
        except ValueError:
            continue
        break
    if dataset_index is not None and feature_index is not None:
        return (
            dataset_index,
            np.take(feature_values[dataset_index], feature_index, axis=1),
        )
    raise KeyError(f"Feature '{feature_name}' not in any dataset.")


def _validate_task_config(task_config: AnalyzeLatentConfig) -> None:
    if "_target_" not in task_config.reducer:
        raise ValueError("Reducer class not specified properly.")


def analyze_latent(config: MOVEConfig) -> None:
    """Train one model to inspect its latent space projections."""

    logger = get_logger(__name__)
    logger.info("Beginning task: analyze latent space")
    task_config = cast(AnalyzeLatentConfig, config.task)
    _validate_task_config(task_config)

    raw_data_path = Path(config.data.raw_data_path)
    interim_path = Path(config.data.interim_data_path)
    output_path = Path(config.data.results_path) / "latent_space"
    output_path.mkdir(exist_ok=True, parents=True)

    logger.debug("Reading data")
    sample_names = io.read_names(raw_data_path / f"{config.data.sample_names}.txt")
    print(f"sample_names length: {len(sample_names)}")
    cat_list, cat_names, con_list, con_names = io.load_preprocessed_data(
        interim_path,
        config.data.categorical_names,
        config.data.continuous_names,
    )

    #19/07/2024 - 17:50 what makes this to be the test dataloader?
    print(f"cat_list.shape: {len(cat_list)}")
    print(f"con_list.shape: {len(con_list)}")
    print(f"cat_list: {cat_list[0].shape}") # (6000, X) # so this is not the test dataset, but the entire dataset
    print(f"con_list: {con_list[0].shape}")
    test_dataloader = make_dataloader(
        cat_list,
        con_list,
        shuffle=False,
        batch_size=task_config.batch_size,
    )
    print(f"Number of samples in test dataset: {len(test_dataloader.dataset)}")
    print(f"len(test_dataloader): {len(test_dataloader)}") # 188
    test_dataset = cast(MOVEDataset, test_dataloader.dataset)
    # print shape of test_dataset
    print(f"test_dataset.cat_all.shape: {test_dataset.cat_all.shape}") # (6000, 328)
    df_index = pd.Index(sample_names, name="sample")

    # print(f"df_index: {df_index}") # 6000 samples15

    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda == True else "cpu")
    model: VAE = hydra.utils.instantiate(
        task_config.model,
        continuous_shapes=test_dataset.con_shapes,
        categorical_shapes=test_dataset.cat_shapes,
    )

    logger.debug(f"Model: {model}")

    model_path = output_path / "model.pt"
    if model_path.exists():
        logger.debug("Re-loading model")
        model.load_state_dict(torch.load(model_path))
        model.to(device)
    else:
        logger.debug("Training model")

        model.to(device)

        # Here it's not the train_dataloader I think, it's the entire dataset?
        # the same function in tune_model.py used the option "split_mask" for the train_dataloader
        train_dataloader = make_dataloader(
            cat_list,
            con_list,
            shuffle=True,
            batch_size=task_config.batch_size,
            drop_last=True,
        )
        # train_dataset = cast(MOVEDataset, train_dataloader.dataset)
        # print(f"train_dataset.cat_all.shape: {train_dataset.cat_all.shape}") # (6000, 328)
        # This is the entire dataset, not the train dataset, but maybe they wanted it like this?
        output: TrainingLoopOutput = hydra.utils.call(
            task_config.training_loop,
            model=model,
            train_dataloader=train_dataloader,
            beta=task_config.model.beta,
        )
        losses = output[:-1]
        torch.save(model.state_dict(), model_path)
        logger.info("Generating visualizations")
        logger.debug("Generating plot: loss curves")
        fig = viz.plot_loss_curves(losses)
        fig_path = str(output_path / "loss_curve.png")
        fig.savefig(fig_path, bbox_inches="tight")
        fig_df = pd.DataFrame(dict(zip(viz.LOSS_LABELS, losses)))
        fig_df.index.name = "epoch"
        fig_df.to_csv(output_path / "loss_curve.tsv", sep="\t")

    model.eval()

    logger.info("Projecting into latent space")
    latent_space = model.project(test_dataloader)
    reducer: TransformerMixin = hydra.utils.instantiate(task_config.reducer)
    embedding = reducer.fit_transform(latent_space)

    mappings_path = interim_path / "mappings.json"
    if mappings_path.exists():
        mappings = io.load_mappings(mappings_path)
    else:
        mappings = {}

    fig_df = pd.DataFrame(
        np.take(embedding, [0, 1], axis=1),
        columns=["dim0", "dim1"],
        index=df_index,
    )

    entire_latent_space_df = pd.DataFrame(latent_space, index=df_index)

    # with open("embeddings.txt", "w") as f:
    #     f.write(f"embedding.shape: {embedding.shape}\n")
    #     f.write(f"embedding.dtype: {embedding.dtype}\n")
    #     f.write(f"embedding[0]: {embedding[0]}\n")
    #     f.write(f"embedding[1]: {embedding[1]}\n")
    #     f.write(f"embedding[2]: {embedding[2]}\n")
    #     f.write(f"embedding[3]: {embedding[3]}\n")
    #     # embedding.shape: (500, 2)
    #     # embedding.dtype: float32
    #     # embedding[0]: [20.939016  -7.5990796]
    #     # embedding[1]: [5.520945  4.6615987]
    #     # embedding[2]: [-6.5492954 11.363279 ]
    #     # embedding[3]: [ 7.593824  -3.3607504]

    # with open("latent_space.txt", "w") as f:
    #     f.write(f"latent_space.shape: {latent_space.shape}\n")
    #     f.write(f"latent_space.dtype: {latent_space.dtype}\n")
    #     f.write(f"type(latent_space): {type(latent_space)}\n")
    #     f.write(f"latent_space[0]: {latent_space[0]}\n")
    #     f.write(f"latent_space[1]: {latent_space[1]}\n")
    #     f.write(f"latent_space[2]: {latent_space[2]}\n")


    for feature_name in task_config.feature_names:
        logger.debug(f"Generating plot: latent space + '{feature_name}'")
        is_categorical = False
        try:
            dataset_index, feature_values = find_feature_values(
                feature_name, cat_names, cat_list
            )
            is_categorical = True
        except KeyError:
            try:
                dataset_index, feature_values = find_feature_values(
                    feature_name, con_names, con_list
                )
            except KeyError:
                logger.warning(f"Feature '{feature_name}' not found in any dataset.")
                continue

        if is_categorical:
            # Convert one-hot encoding to category codes
            is_nan = feature_values.sum(axis=1) == 0
            feature_values = np.argmax(feature_values, axis=1)

            dataset_name = config.data.categorical_names[dataset_index]
            feature_mapping = {
                str(code): category for category, code in mappings[dataset_name].items()
            }
            fig = viz.plot_latent_space_with_cat(
                embedding,
                feature_name,
                feature_values,
                feature_mapping,
                is_nan,
            )
            fig_df[feature_name] = np.where(is_nan, np.nan, feature_values)
            entire_latent_space_df[feature_name] = np.where(is_nan, np.nan, feature_values)
        else:
            feature_values = feature_values
            fig = viz.plot_latent_space_with_con(
                embedding, feature_name, feature_values
            )
            fig_df[feature_name] = np.where(feature_values == 0, np.nan, feature_values)
            entire_latent_space_df[feature_name] = np.where(feature_values == 0, np.nan, feature_values)

        # Remove non-alpha characters
        safe_feature_name = re.sub(r"[^\w\s]", "", feature_name)
        fig_path = str(output_path / f"latent_space_{safe_feature_name}.png")
        fig.savefig(fig_path, bbox_inches="tight")


    entire_latent_space_df.to_csv(output_path / "entire_latent_space.tsv", sep="\t")

    fig_df.to_csv(output_path / "2D_latent_space.tsv", sep="\t")

    logger.info("Reconstructing")
    cat_recons, con_recons = model.reconstruct(test_dataloader)


    con_recons = np.split(con_recons, np.cumsum(model.continuous_shapes[:-1]), axis=1)
    logger.info("Computing reconstruction metrics")
    scores = []
    labels = config.data.categorical_names + config.data.continuous_names
    for cat, cat_recon in zip(cat_list, cat_recons):
        # in tune_model.py it's cat[mask]
        accuracy = calculate_accuracy(cat, cat_recon)
        scores.append(accuracy)
    for con, con_recon in zip(con_list, con_recons):
        cosine_sim = calculate_cosine_similarity(con, con_recon)
        scores.append(cosine_sim)

    logger.debug("Generating plot: reconstruction metrics")

    # Remove the 0s from the scores
    print(f"len(scores): {len(scores)}")
    print(f"scores: {scores}")
    plot_scores = [np.ma.compressed(np.ma.masked_equal(each, np.nan)) for each in scores]
    print(f"len(plot_scores): {len(plot_scores)}")
    print(f"plot_scores: {plot_scores}")


    fig = viz.plot_metrics_boxplot(plot_scores, labels)
    fig_path = str(output_path / "reconstruction_metrics.png")
    fig.savefig(fig_path, bbox_inches="tight")
    fig_df = pd.DataFrame(dict(zip(labels, scores)), index=df_index)
    fig_df = fig_df.replace(np.nan, "NA")
    fig_df.to_csv(output_path / "reconstruction_metrics.tsv", sep="\t")

    logger.info("Computing feature importance")
    num_samples = len(cast(Sized, test_dataloader.sampler))
    for i, dataset_name in enumerate(config.data.categorical_names):
        logger.debug(f"Generating plot: feature importance '{dataset_name}'")
        na_value = one_hot_encode_single(mappings[dataset_name], None)
        dataloaders = perturb_categorical_data(
            test_dataloader, config.data.categorical_names, dataset_name, na_value
        )

        num_features = len(dataloaders)
        print(f"num_features of categorical dataset {dataset_name}: {num_features}")
        z = model.project(test_dataloader)
        diffs = np.empty((num_samples, num_features))
        for j, dataloader in enumerate(dataloaders):
            z_perturb = model.project(dataloader)
            diffs[:, j] = np.sum(z_perturb - z, axis=1)
        feature_mapping = {
            str(code): category for category, code in mappings[dataset_name].items()
        }
        fig = viz.plot_categorical_feature_importance(
            diffs, cat_list[i], cat_names[i], feature_mapping
        )
        fig_path = str(output_path / f"feat_importance_{dataset_name}.png")
        fig.savefig(fig_path, bbox_inches="tight")
        fig_df = pd.DataFrame(diffs, columns=cat_names[i], index=df_index)
        fig_df.to_csv(output_path / f"feat_importance_{dataset_name}.tsv", sep="\t")

    for i, dataset_name in enumerate(config.data.continuous_names):
        logger.debug(f"Generating plot: feature importance '{dataset_name}'")
        print(f"Generating plot: feature importance '{dataset_name}'")
        dataloaders = perturb_continuous_data(
            test_dataloader, config.data.continuous_names, dataset_name, 0.0
        )
        num_features = len(dataloaders)
        # print(f"num_features of continuous dataset {dataset_name}: {num_features}")
        # print(f"i: {i}")
        # print(f"type(i): {type(i)}")
        z = model.project(test_dataloader)
        diffs = np.empty((num_samples, num_features))
        for j, dataloader in enumerate(dataloaders):
            z_perturb = model.project(dataloader)
            diffs[:, j] = np.sum(z_perturb - z, axis=1)
        fig = viz.plot_continuous_feature_importance(diffs, con_list[i], con_names[i])
        fig_path = str(output_path / f"feat_importance_{dataset_name}.png")
        fig.savefig(fig_path, bbox_inches="tight")
        fig_df = pd.DataFrame(diffs, columns=con_names[i], index=df_index)
        fig_df.to_csv(output_path / f"feat_importance_{dataset_name}.tsv", sep="\t")
