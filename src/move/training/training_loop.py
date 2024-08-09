from typing import Optional

from torch.utils.data import DataLoader

from move.models.vae import VAE

import pandas as pd
TrainingLoopOutput = tuple[list[float], list[float], list[float], list[float], float]

from move.core.logging import get_logger
logger = get_logger(__name__)
def dilate_batch(dataloader: DataLoader) -> DataLoader:
    """
    Increase the batch size of a dataloader.

    Args:
        dataloader (DataLoader): An object feeding data to the VAE

    Returns:
        DataLoader: An object feeding data to the VAE
    """
    assert dataloader.batch_size is not None
    dataset = dataloader.dataset
    batch_size = int(dataloader.batch_size * 1.5)
    return DataLoader(dataset, batch_size, shuffle=True, drop_last=True)

def cyclical_annealing(epoch, num_epochs, cycles=4, max_beta=1.0):
    """
    Compute the cyclical annealing value for KLD weight.

    Args:
        epoch (int): Current epoch.
        num_epochs (int): Total number of epochs.
        cycles (int): Number of cycles in the annealing schedule.
        max_beta (float): Maximum value for beta (KLD weight).

    Returns:
        float: The KLD weight for the current epoch.
    """
    period = num_epochs / cycles # 100 / 4 = 25
    epoch_in_cycle = epoch % period # 1 % 25 = 1
    beta = min(max_beta, (epoch_in_cycle / period) * max_beta)
    return beta

def training_loop(
    model: VAE,
    train_dataloader: DataLoader,
    valid_dataloader: Optional[DataLoader] = None,
    lr: float = 1e-4,
    num_epochs: int = 100,
    batch_dilation_steps: list[int] = [],
    kld_warmup_steps: list[int] = [],
    early_stopping: bool = False,
    patience: int = 0,
    # beta: float = 1,
    # num_hidden: list[int] = [120,120],
) -> TrainingLoopOutput:
    """
    Trains a VAE model with batch dilation and KLD warm-up. Optionally,
    enforce early stopping.

    Args:
        model (VAE): trained VAE model object
        train_dataloader (DataLoader):  An object feeding data to the VAE with training data
        valid_dataloader (Optional[DataLoader], optional): An object feeding data to the VAE with validation data. Defaults to None.
        lr (float, optional): learning rate. Defaults to 1e-4.
        num_epochs (int, optional): number of epochs. Defaults to 100.
        batch_dilation_steps (list[int], optional): a list with integers corresponding to epochs when batch size is increased. Defaults to [].
        kld_warmup_steps (list[int], optional):  a list with integers corresponding to epochs when kld is decreased by the selected rate. Defaults to [].
        early_stopping (bool, optional):  boolean if use early stopping . Defaults to False.
        patience (int, optional): number of epochs to wait before early stop if no progress on the validation set . Defaults to 0.

    Returns:
        (tuple): a tuple containing:
            *outputs (*list): lists containing information of epoch loss, BCE loss, SSE loss, KLD loss
            kld_weight (float): final KLD after dilations during the training
    """

    outputs = [[] for _ in range(4)]
    min_likelihood = float("inf")
    counter = 0

    cycles = 4
    kld_weight = 0

    target_KLD_weight = model.beta
    # Original code
    # target_KLD_weight = beta * (num_latent**-1)

    # Removing latent dimension from the KLD weight
    # target_KLD_weight = beta
    # increment = target_KLD_weight / len(kld_warmup_steps)



    warmup_log = []
    for epoch in range(1, num_epochs + 1):
        if epoch in kld_warmup_steps:

            kld_weight += increment  # Increment kld_multiplier

            print(f"Epoch {epoch} - Target KLD weight: {target_KLD_weight} - KLD weight: {kld_weight}")
            warmup_log.append({
            "epoch": epoch,
            "kld_weight": kld_weight,
            "target_KLD_weight": target_KLD_weight,
            "increment" : increment,
            # "kld_multiplier": kld_multiplier,
            # "num_latent": num_latent,
            # "num_hidden": num_hidden,
            
        })           
    # Calculate the cyclical KLD weight for the current epoch
        # kld_weight = cyclical_annealing(epoch, num_epochs, cycles, target_KLD_weight)

        # print(f"Epoch {epoch} - KLD weight: {kld_weight}")

        warmup_log.append({
            "epoch": epoch,
            "kld_weight": kld_weight,
            "target_KLD_weight": target_KLD_weight,
            "cycles": cycles,
        }) 

        if epoch in batch_dilation_steps:
            train_dataloader = dilate_batch(train_dataloader)

        for i, output in enumerate(
            model.encoding(train_dataloader, epoch, lr, kld_weight)
        ):
            outputs[i].append(output)

        if early_stopping and valid_dataloader is not None:
            output = model.latent(valid_dataloader, kld_weight)
            valid_likelihood = output[-1]
            if valid_likelihood > min_likelihood and counter < patience:
                counter += 1
                if counter % 5 == 0:
                    lr *= 0.9
            elif counter == patience:
                break
            else:
                min_likelihood = valid_likelihood
                counter = 0


    
    # model_path = "model.pt"
    # summary = summary(model, (1, 120))
    # print(str(model))
    # print(f"Printing model")
    # print(model)


    warmup_df = pd.DataFrame(warmup_log)
    # outputfile = "/faststorage/jail/project/igpv/SCZ-RWE/results/2024-06-08-MOVE_trial/kldw_warmup_trial/warmup_log.csv"

    # warmup_df.to_csv("/faststorage/jail/project/igpv/SCZ-RWE/results/2024-06-08-MOVE_trial/kldw_warmup_trial/warmup_log.csv", index=False)
    warmup_df.to_csv("warmup_log.csv", index=False)

    return *outputs, kld_weight
