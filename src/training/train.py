import time
import torch
import numpy as np
import torch.nn as nn

from tqdm import tqdm  # noqa
from torchcontrib.optim import SWA
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from torch.utils.data.sampler import RandomSampler
from transformers import get_linear_schedule_with_warmup

from params import NUM_WORKERS
from training.meter import NFLMeter
from model_zoo.models_det import get_val_model
from training.optim import define_optimizer
from training.sampler import PlayerSampler


def collate_fn(batch):
    return tuple(zip(*batch))


def fit_det(
    model,
    train_dataset,
    val_dataset,
    optimizer_name="adam",
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    acc_steps=1,
    swa_first_epoch=50,
    num_classes=1,
    verbose=1,
    first_epoch_eval=0,
    device="cuda",
):
    """
    Fitting function for the object detection task.

    Args:
        model (torch model): Model to train.
        train_dataset (torch dataset): Dataset to train with.
        val_dataset (torch dataset): Dataset to validate with.
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        epochs (int, optional): Number of epochs. Defaults to 50.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        warmup_prop (float, optional): Warmup proportion. Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        acc_steps (int, optional): Accumulation steps. Defaults to 1.
        swa_first_epoch (int, optional): Epoch to start applying SWA from. Defaults to 50.
        verbose (int, optional): Period (in epochs) to display logs at. Defaults to 1.
        first_epoch_eval (int, optional): Epoch to start evaluating at. Defaults to 0.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        NFLMeter: Meter with predictions.
    """

    optimizer = define_optimizer(optimizer_name, model.parameters(), lr=lr)

    if swa_first_epoch <= epochs:
        optimizer = SWA(optimizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_training_steps = int(epochs * len(train_loader) / acc_steps)
    num_warmup_steps = int(warmup_prop * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()

        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0

        if epoch + 1 > swa_first_epoch:
            optimizer.swap_swa_sgd()
            # print("Swap to SGD")

        for step, batch in enumerate(train_loader):
            images = torch.stack(batch[0]).to(device)
            boxes = [box.to(device) for box in batch[1]]
            labels = [label.to(device) for label in batch[2]]

            loss, _, _ = model(images, boxes, labels)
            loss.backward()

            avg_loss += loss.item() / len(train_loader)

            if ((step + 1) % acc_steps) == 0:
                optimizer.step()
                scheduler.step()
                for param in model.parameters():
                    param.grad = None

        if epoch + 1 >= swa_first_epoch:
            # print("update + swap to SWA")
            optimizer.update_swa()
            optimizer.swap_swa_sgd()

        meter = NFLMeter(num_classes=num_classes)

        if epoch + 1 >= first_epoch_eval or epoch + 1 == epochs:
            model_eval = get_val_model(model).to(device)
            with torch.no_grad():
                for batch in val_loader:
                    images = torch.stack(batch[0]).to(device)
                    scales = torch.tensor([1] * len(images)).long().to(device)

                    y_pred = model_eval(images, scales).detach()

                    meter.update(batch, y_pred)

            score_1, score_2, score_3 = meter.compute_scores()

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s\t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )
            if epoch + 1 >= first_epoch_eval:
                print(f"scores : {score_1:.4f} - {score_2:.4f} - {score_3:.4f}")
            else:
                print("")

    del val_loader, train_loader, y_pred
    torch.cuda.empty_cache()

    return meter


def fit_cls(
    model,
    train_dataset,
    val_dataset,
    optimizer_name="adam",
    samples_per_player=0,
    epochs=50,
    batch_size=32,
    val_bs=32,
    warmup_prop=0.1,
    lr=1e-3,
    acc_steps=1,
    swa_first_epoch=50,
    num_classes_aux=0,
    aux_mode="sigmoid",
    verbose=1,
    first_epoch_eval=0,
    device="cuda",
):
    """
    Fitting function for the classification task.

    Args:
        model (torch model): Model to train.
        train_dataset (torch dataset): Dataset to train with.
        val_dataset (torch dataset): Dataset to validate with.
        optimizer_name (str, optional): Optimizer name. Defaults to 'adam'.
        samples_per_player (int, optional): Number of images to use per player. Defaults to 0.
        epochs (int, optional): Number of epochs. Defaults to 50.
        batch_size (int, optional): Training batch size. Defaults to 32.
        val_bs (int, optional): Validation batch size. Defaults to 32.
        warmup_prop (float, optional): Warmup proportion. Defaults to 0.1.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        acc_steps (int, optional): Accumulation steps. Defaults to 1.
        swa_first_epoch (int, optional): Epoch to start applying SWA from. Defaults to 50.
        verbose (int, optional): Period (in epochs) to display logs at. Defaults to 1.
        first_epoch_eval (int, optional): Epoch to start evaluating at. Defaults to 0.
        device (str, optional): Device for torch. Defaults to "cuda".

    Returns:
        numpy array [len(val_dataset)]: Last predictions on the validation data.
    """

    optimizer = define_optimizer(optimizer_name, model.parameters(), lr=lr)

    if swa_first_epoch <= epochs:
        optimizer = SWA(optimizer)

    loss_fct = nn.BCEWithLogitsLoss()
    loss_fct_aux = nn.BCEWithLogitsLoss() if aux_mode == "sigmoid" else nn.CrossEntropyLoss()
    aux_loss_weight = 1 if num_classes_aux else 0

    if samples_per_player:
        sampler = PlayerSampler(
            RandomSampler(train_dataset),
            train_dataset.players,
            batch_size=batch_size,
            drop_last=True,
            samples_per_player=samples_per_player,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        print(
            f"Using {len(train_loader)} out of {len(train_dataset) // batch_size} "
            f"batches by limiting to {samples_per_player} samples per player.\n"
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    num_training_steps = int(epochs * len(train_loader))
    num_warmup_steps = int(warmup_prop * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps, num_training_steps
    )

    for epoch in range(epochs):
        model.train()

        start_time = time.time()
        optimizer.zero_grad()

        avg_loss = 0

        if epoch + 1 > swa_first_epoch:
            optimizer.swap_swa_sgd()
            # print("Swap to SGD")

        for batch in train_loader:
            images = batch[0].to(device)
            y_batch = batch[1].to(device).view(-1).float()
            y_batch_aux = batch[2].to(device).float()
            y_batch_aux = y_batch_aux.float() if aux_mode == "sigmoid" else y_batch_aux.long()

            y_pred, y_pred_aux = model(images)

            loss = loss_fct(y_pred.view(-1), y_batch)
            if aux_loss_weight:
                loss += aux_loss_weight * loss_fct_aux(y_pred_aux, y_batch_aux)
            loss.backward()

            avg_loss += loss.item() / len(train_loader)
            optimizer.step()
            scheduler.step()
            for param in model.parameters():
                param.grad = None

        if epoch + 1 >= swa_first_epoch:
            # print("update + swap to SWA")
            optimizer.update_swa()
            optimizer.swap_swa_sgd()

        preds = np.empty(0)
        preds_aux = np.empty((0, num_classes_aux))
        model.eval()
        avg_val_loss = 0.0
        if epoch + 1 >= first_epoch_eval or epoch + 1 == epochs:
            with torch.no_grad():
                for batch in val_loader:
                    images = batch[0].to(device)
                    y_batch = batch[1].to(device).view(-1).float()
                    y_aux = batch[2].to(device).float()
                    y_batch_aux = y_aux.float() if aux_mode == "sigmoid" else y_aux.long()

                    y_pred, y_pred_aux = model(images)

                    loss = loss_fct(y_pred.detach().view(-1), y_batch)
                    if aux_loss_weight:
                        loss += aux_loss_weight * loss_fct_aux(
                            y_pred_aux.detach(), y_batch_aux
                        )

                    avg_val_loss += loss.item() / len(val_loader)

                    y_pred = torch.sigmoid(y_pred).view(-1)
                    y_pred_aux = (
                        y_pred_aux.sigmoid() if aux_mode == "sigmoid"
                        else y_pred_aux.softmax(-1)
                    )

                    preds = np.concatenate([preds, y_pred.detach().cpu().numpy()])
                    preds_aux = np.concatenate(
                        [preds_aux, y_pred_aux.detach().cpu().numpy()]
                    )

        auc = roc_auc_score(val_dataset.labels, preds)

        if aux_mode == "sigmoid":
            scores_aux = np.round([
                roc_auc_score(val_dataset.aux_labels[:, i], preds_aux[:, i])
                for i in range(num_classes_aux)
            ], 3,).tolist()
        else:
            scores_aux = np.round([
                roc_auc_score((val_dataset.aux_labels == i).astype(int), preds_aux[:, i])
                for i in range(num_classes_aux)
            ], 3,).tolist()

        elapsed_time = time.time() - start_time
        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} \t lr={lr:.1e}\t t={elapsed_time:.0f}s \t"
                f"loss={avg_loss:.3f}",
                end="\t",
            )

            if epoch + 1 >= first_epoch_eval:
                print(
                    f"val_loss={avg_val_loss:.3f} \t auc={auc:.3f}\t aucs_aux={scores_aux}"
                )
            else:
                print("")

    del val_loader, train_loader, y_pred
    torch.cuda.empty_cache()

    return preds, preds_aux
