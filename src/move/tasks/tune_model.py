__all__ = ["tune_model"]

from pathlib import Path
from random import shuffle
from typing import Any, Literal, cast
import re

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from matplotlib.cbook import boxplot_stats
from numpy.typing import ArrayLike
from omegaconf import OmegaConf
import move.visualization as viz
from sklearn.metrics.pairwise import cosine_similarity

from move.analysis.metrics import (
    calculate_accuracy,
    calculate_cosine_similarity,
    calculate_mse_rmse,
)
from move.conf.schema import (
    MOVEConfig,
    TuneModelConfig,
    TuneModelReconstructionConfig,
    TuneModelStabilityConfig,
)
from move.core.logging import get_logger
from move.core.typing import BoolArray, FloatArray
from move.data import io
from move.data.dataloaders import MOVEDataset, make_dataloader, split_samples
from move.models.vae import VAE


TaskType = Literal["reconstruction", "stability"]


def _get_task_type(
    task_config: TuneModelConfig,
) -> TaskType:
    task_type = OmegaConf.get_type(task_config)
    if task_type is TuneModelReconstructionConfig:
        return "reconstruction"
    if task_type is TuneModelStabilityConfig:
        return "stability"
    raise ValueError("Unsupported type of task!")


def _get_record(values: ArrayLike, **kwargs) -> dict[str, Any]:
    record = kwargs
    bxp_stats, *_ = boxplot_stats(values)
    bxp_stats.pop("fliers")
    record.update(bxp_stats)
    return record


def tune_model(config: MOVEConfig) -> float:
    """Train multiple models to tune the model hyperparameters."""
    hydra_config = HydraConfig.get()

    if hydra_config.mode != RunMode.MULTIRUN:
        raise ValueError("This task must run in multirun mode.")

    # Delete sweep run config
    sweep_config_path = Path(hydra_config.sweep.dir).joinpath("multirun.yaml")
    if sweep_config_path.exists():
        sweep_config_path.unlink()

    job_num = hydra_config.job.num + 1

    logger = get_logger(__name__)
    task_config = cast(TuneModelConfig, config.task)
    task_type = _get_task_type(task_config)

    logger.info(f"Beginning task: tune model {task_type} {job_num}")
    logger.info(f"Job name: {hydra_config.job.override_dirname}")

    interim_path = Path(config.data.interim_data_path)
    output_path = Path(config.data.results_path) / "tune_model"
    output_path.mkdir(exist_ok=True, parents=True)

    logger.debug("Reading data")

    cat_list, _, con_list, _ = io.load_preprocessed_data(
        interim_path,
        config.data.categorical_names,
        config.data.continuous_names,
    )
    # print(f"Total number of samples in the cat_list: {len(cat_list[0])} \nTotal number of samples in the con_list: {len(con_list[0])}")


    assert task_config.model is not None
    device = torch.device("cuda" if task_config.model.cuda == True else "cpu")

    def _tune_stability(
        task_config: TuneModelStabilityConfig,
    ):
        label = [hp.split("=") for hp in hydra_config.job.override_dirname.split(",")]

        train_dataloader = make_dataloader(
            cat_list,
            con_list,
            shuffle=True,
            batch_size=task_config.batch_size,
            drop_last=True,
        )

        test_dataloader = make_dataloader(
            cat_list,
            con_list,
            shuffle=False,
            batch_size=task_config.batch_size,
            drop_last=False,
        )

        train_dataset = cast(MOVEDataset, train_dataloader.dataset)

        logger.info(f"Training {task_config.num_refits} refits")

        cosine_sim0 = None
        cosine_sim_diffs = []
        for j in range(task_config.num_refits):
            logger.debug(f"Refit: {j+1}/{task_config.num_refits}")
            model: VAE = hydra.utils.instantiate(
                task_config.model,
                continuous_shapes=train_dataset.con_shapes,
                categorical_shapes=train_dataset.cat_shapes,
            )
            model.to(device)

            hydra.utils.call(
            task_config.training_loop,
            model=model,
            train_dataloader=train_dataloader,
            # beta=task_config.model.beta
            )

            model.eval()
            
            latent, *_ = model.latent(test_dataloader, kld_weight=model.beta) 
            # Here it doesn't matter what KLD you use. We don't care about the loss function, I don't need that for calculating the latent spaces

            # Leo version
            # latent, *_, test_likelihood = model.latent(test_dataloader, kld_weight=model.beta)
            # But I want the test_likelihood in the reconstruction, while doing hyperparameter tuning! 

            if cosine_sim0 is None:
                cosine_sim0 = cosine_similarity(latent)
            else:
                cosine_sim = cosine_similarity(latent)
                D = np.absolute(cosine_sim - cosine_sim0)
                # removing the diagonal element (cos_sim with itself)
                diff = D[~np.eye(D.shape[0], dtype=bool)].reshape(D.shape[0], -1)
                mean_diff = np.mean(diff)
                cosine_sim_diffs.append(mean_diff)

        record = _get_record(
            cosine_sim_diffs,
            job_num=job_num,
            **dict(label),
            metric="mean_diff_cosine_similarity",
            num_refits=task_config.num_refits,
        )
        logger.info("Writing results")
        df_path = output_path / "stability_stats.tsv"
        header = not df_path.exists()
        df = pd.DataFrame.from_records([record])
        df.to_csv(df_path, sep="\t", mode="a", header=header, index=False)

    def _tune_reconstruction(
        task_config: TuneModelReconstructionConfig,
    ):
        split_path = interim_path / "split_mask.npy"
        scores_folder = output_path / "reconstruction_scores"
        #create the directory
        scores_folder.mkdir(exist_ok=True, parents=True)

        if split_path.exists():
            split_mask: BoolArray = np.load(split_path)
        else:
            # raise error
            raise ValueError("Split mask not found")
            # num_samples = cat_list[0].shape[0] if cat_list else con_list[0].shape[0]
            # split_mask = split_samples(num_samples, 0.9)
            # np.save(split_path, split_mask)

        # print(f"cat_list[0].shape[0]: {cat_list}")
        # print(f"con_list[0].shape[0]: {con_list}")
        train_dataloader = make_dataloader(
            cat_list,
            con_list,
            split_mask,
            shuffle=True,
            batch_size=task_config.batch_size,
            drop_last=True,
        )

        # print(f"Number of samples in train dataset in tune_reconstruction: {len(train_dataloader.dataset)}")
        # 4800
        # print(f"Number of samples in test dataset: {len(test_dataloader.dataset)}")



        train_dataset = cast(MOVEDataset, train_dataloader.dataset)

        model: VAE = hydra.utils.instantiate(
            task_config.model,
            continuous_shapes=train_dataset.con_shapes,
            categorical_shapes=train_dataset.cat_shapes,
            num_hidden=task_config.model.num_hidden,
        )
        model.to(device)
        # beta = task_config.model.beta

        logger.debug(f"Model: {model}")

        logger.debug("Training model")
        hydra.utils.call(
            task_config.training_loop,
            model=model,
            train_dataloader=train_dataloader,
            # beta=task_config.model.beta,
        )

        # output: TrainingLoopOutput = hydra.utils.call(
        #     task_config.training_loop,
        #     model=model,
        #     train_dataloader=train_dataloader,
        #     beta=task_config.model.beta,
        #     num_latent=task_config.model.num_latent
        # )

        # Added 14/07/2024 - 11:15 to save the loss curves
        # losses = output[:-3]
        # torch.save(model.state_dict(), model_path)
        # logger.info("Generating visualizations")
        # logger.debug("Generating plot: loss curves")
        # fig = viz.plot_loss_curves(losses)
        # # job_num = hydra_config.job.num + 1
        # fig_path = str(output_path / f"loss_curve_{job_num}.png")
        # fig.savefig(fig_path, bbox_inches="tight")
        # fig_df = pd.DataFrame(dict(zip(viz.LOSS_LABELS, losses)))
        # fig_df.index.name = "epoch"
        # fig_df.to_csv(output_path / f"loss_curve_{job_num}.tsv", sep="\t")
        # # save output to tsv file
        # output_df = pd.DataFrame(output, columns=["epoch_loss", "bce_loss", "sse_loss", "kld_loss"])


        model.eval()
        logger.info("Reconstructing")
        logger.info("Computing reconstruction metrics")
        label = [hp.split("=") for hp in hydra_config.job.override_dirname.split(",")]


        pattern = r'([^,=]+)=(\[[^\]]*\]|[^,]+)'

        # Find all matches
        matches = re.findall(pattern, hydra_config.job.override_dirname)

        # print(matches)

        # convert the matches into a list of list
        label = [list(match) for match in matches]
        # print(f"hydra_config.job.override_dirname: {hydra_config.job.override_dirname}")

        # label2 = [hp for hp in hydra_config.job.override_dirname.split(",")]
        # print(f"label2")
        # print(f"label: {label}")

        # print(f"label: {label}")
        # print each key-value pair in the label
        # print each element with their index
        # for i in range(len(label)):
        #     print(f"index: {i}, element: {label[i]}")
        # for key, value in label:
        #     print(f"key: {key}, value: {value}")
        records = []
        # records_test_likelihood = []
        df_test_metrics = pd.DataFrame()
        splits = zip(["train", "test"], [split_mask, ~split_mask])

        # logger.info("for split_name, mask in splits:")

        for split_name, mask in splits:
            dataloader = make_dataloader(
                cat_list,
                con_list,
                mask,
                shuffle=False,
                batch_size=task_config.batch_size,
            )
            cat_recons, con_recons = model.reconstruct(dataloader)


            # record = _get_record(
            #     accuracy,
            #     job_num=job_num,
            #     **dict(label),
            #     metric="accuracy",
            #     dataset=dataset_name,
            #     split=split_name,
            # )
            # records.append(record)
            # if mask is test, I can get the test_likelihood
            # logger.info(f"split_name: {split_name}")
            if split_name == "test":
                latent, *_, test_likelihood, test_kld = model.latent(dataloader, kld_weight=model.beta)
                # convert test_likelihood to number
                test_likelihood = test_likelihood.item()

                logger.info("Before test_kld")
                # test_kld = test_kld.item()
                logger.info(f"test_kld: {test_kld}")
                logger.info("After test_kld")
                
                label_dict = {key: value for key, value in label}
                

# 

                df_test_tmp = pd.DataFrame([{"job_num": job_num, **label_dict, "test_likelihood": test_likelihood, "test_kld": test_kld}])

                df_test_metrics = pd.concat([df_test_metrics, df_test_tmp])


            
            con_recons = np.split(con_recons, np.cumsum(model.continuous_shapes[:-1]), axis=1)


            scores = []
            mse_scores = []
            rmse_scores = []
            labels = config.data.categorical_names + config.data.continuous_names
            for cat, cat_recon, dataset_name in zip(
                cat_list, cat_recons, config.data.categorical_names
            ):

                # print(f"cat: {cat}")
                # print(f"cat.shape: {cat.shape}")
                # print(f"cat[mask]: {cat[mask]}")
                # print(f"cat[mask].shape: {cat[mask].shape}")
                # print(f"cat_recon: {cat_recon}")
                # print(f"cat_recon.shape: {cat_recon.shape}")
                logger.debug(f"Computing accuracy: '{dataset_name}'")
                accuracy = calculate_accuracy(cat[mask], cat_recon)
                scores.append(accuracy)
                record = _get_record(
                    accuracy,
                    job_num=job_num,
                    **dict(label),
                    metric="accuracy",
                    dataset=dataset_name,
                    split=split_name,
                )
                records.append(record)
            for con, con_recon, dataset_name in zip(
                con_list, con_recons, config.data.continuous_names
            ):
                logger.debug(f"Computing cosine similarity: '{dataset_name}'")
                # print(f"con: {con}")
                # print(f"con.shape: {con.shape}")
                # print(f"con[mask]: {con[mask]}")
                # print(f"con[mask].shape: {con[mask].shape}")
                # print(f"con_recon: {con_recon}")
                # print(f"con_recon.shape: {con_recon.shape}")

                cosine_sim = calculate_cosine_similarity(con[mask], con_recon)
                mse, rmse = calculate_mse_rmse(con[mask], con_recon)


                # print(f"{dataset_name} size cosine_sim BEFORE removing the 0: {cosine_sim.shape}")
                # print(f"cosine_sim: {cosine_sim}")

                # cosine_sim is a list. Remove the values that are 0
                # cosine_sim = [i for i in cosine_sim if i != -9]
                # mse = [i for i in mse if i != -9]


                cosine_sim_for_record = [i for i in cosine_sim if not np.isnan(i)]
                mse_for_record = [i for i in mse if not np.isnan(i)]
                rmse_for_record = [i for i in rmse if not np.isnan(i)]

                # I cannot append these to the scores and then create the dataframe because it expects all the samples
                scores.append(cosine_sim)
                mse_scores.append(mse)
                rmse_scores.append(rmse)
                # cosine_sim = [np.ma.compressed(np.ma.masked_equal(each, 0)) for each in cosine_sim]
                # print(f"{dataset_name} size cosine_sim AFTER removing the 0: {len(cosine_sim_for_record)}")
                # print(f"cosine_sim: {cosine_sim}")

                record = _get_record(
                    cosine_sim_for_record,
                    job_num=job_num,
                    **dict(label),
                    metric="cosine_similarity",
                    dataset=dataset_name,
                    split=split_name,
                )
                records.append(record)

                record_mse = _get_record(
                    mse_for_record,
                    job_num=job_num,
                    **dict(label),
                    metric="mse",
                    dataset=dataset_name,
                    split=split_name,
                )

                records.append(record_mse)

                record_rmse = _get_record(
                    rmse_for_record,
                    job_num=job_num,
                    **dict(label),
                    metric="rmse",
                    dataset=dataset_name,
                    split=split_name,
                )

                records.append(record_rmse)

            raw_data_path = Path(config.data.raw_data_path)
            # sample_names = io.read_names(raw_data_path / f"{config.data.sample_names}.txt")
            train_test_splits_file_name = Path(config.data.train_test_splits_file_name)
            train_test_path = raw_data_path / train_test_splits_file_name
            # print(train_test_path)

            train_test_splits = pd.read_csv(train_test_path, sep = "\t")
            # sample_names_df = train_test_splits["Split"] == split_name
            sample_names_df = train_test_splits.query("Split == @split_name")
            sample_names = sample_names_df["ID"].tolist()
            # print(f"{split_name} len(sample_names): {len(sample_names)}")

            df_index = pd.Index(sample_names, name="sample")
            scores_df = pd.DataFrame(dict(zip(labels, scores)), index=df_index)
            # replace np.nan with NA
            scores_df = scores_df.replace(np.nan, "NA")

            # length of values (4991) doesn't match length of index (8799)
            # 8799 is the training set
            # 4991 is the age_mental_behav size cosine_sim after removing the 0s
            scores_df.to_csv(scores_folder / f"{job_num}_{split_name}_reconstruction_scores.tsv", sep="\t")

            mse_scores_df = pd.DataFrame(dict(zip(config.data.continuous_names, mse_scores)), index=df_index)
            mse_scores_df = mse_scores_df.replace(np.nan, "NA")
            mse_scores_df.to_csv(scores_folder / f"{job_num}_{split_name}_reconstruction_mse_scores.tsv", sep="\t")

            rmse_scores_df = pd.DataFrame(dict(zip(config.data.continuous_names, rmse_scores)), index=df_index)
            rmse_scores_df = rmse_scores_df.replace(np.nan, "NA")
            rmse_scores_df.to_csv(scores_folder / f"{job_num}_{split_name}_reconstruction_rmse_scores.tsv", sep="\t")

            # Maybe the bad performance of age at diagnosis in tune_reconstruction and not in analyze_latent is due to the fact that the scores get put to 0 in the analyze_latent function?
            # fig = viz.plot_metrics_boxplot(scores, labels)
            # fig_path = str(output_path / f"{job_num}_{split_name}_nozeroscore_reconstruction_metrics.png")
            # fig.savefig(fig_path, bbox_inches="tight")

            # plot_scores = [np.ma.compressed(np.ma.masked_equal(each, 0)) for each in scores]
            # fig = viz.plot_metrics_boxplot(plot_scores, labels)
            # fig_path = str(output_path / f"{job_num}_{split_name}_zeroscore_reconstruction_metrics.png")
            # fig.savefig(fig_path, bbox_inches="tight")



        logger.info("Writing results")
        df_path = output_path / "reconstruction_stats.tsv"
        header = not df_path.exists()
        df = pd.DataFrame.from_records(records)
        df.to_csv(df_path, sep="\t", mode="a", header=header, index=False)


        structures_path = output_path / "models_structures"
        structures_path.mkdir(exist_ok=True, parents=True)
        with open(structures_path / f"{job_num}_model_structure.txt", "w") as f:
            f.write(str(model))


        # df_test_metrics = pd.DataFrame.from_records(records_test_likelihood)
        df_test_metrics.to_csv(output_path / "test_metrics.tsv", sep="\t", mode="a", header=header, index=False)


    if task_type == "reconstruction":
        task_config = cast(TuneModelReconstructionConfig, task_config)
        _tune_reconstruction(task_config)
    elif task_type == "stability":
        task_config = cast(TuneModelStabilityConfig, task_config)
        _tune_stability(task_config)

    return 0.0


# Key num_refits not in TuneModelReconstructionConfig