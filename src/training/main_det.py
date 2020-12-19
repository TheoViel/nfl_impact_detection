import gc
import torch
import numpy as np

from sklearn.model_selection import GroupKFold

from training.train import fit_det
from utils.save import save_pickle
from data.dataset import NFLDatasetDet
from data.transforms import get_transfos_det
from model_zoo.models_det import get_model
from utils.torch import seed_everything, count_parameters, save_model_weights


def train_det(config, df_train, df_val, fold, log_folder=None):
    """
    Trains and validate a model.

    Args:
        config (Config): Parameters.
        df_train (pandas dataframe): Training metadata.
        df_val (pandas dataframe): Validation metadata.
        fold (int): Selected fold.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.

    Returns:
        np array: Validation predictions.
        pandas dataframe: Training history.
    """
    seed_everything(config.seed)

    if config.num_classes == 1:
        df_train['impact'] = 1
        df_train['extended_impact'] = 1

    model = get_model(
        config.name,
        num_classes=config.num_classes,
    ).to(config.device)

    model.zero_grad()

    train_dataset = NFLDatasetDet(
        df_train.copy(),
        transforms=get_transfos_det(train=True),
        root=config.img_path,
        train=True,
    )

    val_dataset = NFLDatasetDet(
        df_val.copy(),
        transforms=get_transfos_det(train=False),
        root=config.img_path,
    )

    n_parameters = count_parameters(model)

    print(f"    -> {len(train_dataset)} training images")
    print(f"    -> {len(val_dataset)} validation images")
    print(f"    -> {n_parameters} trainable parameters\n")

    meter = fit_det(
        model,
        train_dataset,
        val_dataset,
        optimizer_name=config.optimizer,
        epochs=config.epochs,
        batch_size=config.batch_size,
        val_bs=config.val_bs,
        lr=config.lr,
        acc_steps=config.acc_steps,
        warmup_prop=config.warmup_prop,
        swa_first_epoch=config.swa_first_epoch,
        num_classes=config.num_classes,
        verbose=config.verbose,
        first_epoch_eval=config.first_epoch_eval,
        device=config.device,
    )

    if config.save_weights and log_folder is not None:
        save_model_weights(
            model,
            f"{config.name}_{fold}.pt",
            cp_folder=log_folder,
        )

    del model, train_dataset, val_dataset
    torch.cuda.empty_cache()

    return meter


def k_fold_det(config, df, log_folder=None):
    """
    Performs a video grouped k-fold cross validation for the deteciton task.

    Args:
        config (Config): Parameters.
        df (pandas dataframe): Metadata.
        log_folder (None or str, optional): Folder to logs results to. Defaults to None.
    """

    if "val_idx" not in df.columns:
        gkf = GroupKFold(n_splits=config.k)
        splits = list(gkf.split(X=df, y=df, groups=df["gameKey"]))
    else:
        idx = np.arange(len(df))
        splits = [(idx[df['val_idx'] != k], idx[df['val_idx'] == k]) for k in range(config.k)]

    for i, (train_idx, val_idx) in enumerate(splits):
        if i in config.selected_folds:
            print(f"\n-------------   Fold {i + 1} / {config.k}  -------------\n")

            df_train = df.iloc[train_idx].copy()
            df_val = df.iloc[val_idx].copy()

            meter = train_det(config, df_train, df_val, i, log_folder=log_folder)
            meter.val_idx = val_idx

            if log_folder is not None:
                save_pickle(meter, log_folder + f"meter_{i}.pkl")
                del meter
                gc.collect()

            if log_folder is None and len(config.selected_folds) == 1:
                return meter
