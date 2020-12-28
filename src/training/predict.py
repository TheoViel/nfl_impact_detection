import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import NFLDatasetCls
from data.transforms import get_transfos_cls
from params import NUM_WORKERS


def predict_cls(
    model,
    df,
    batch_size=256,
    device="cuda",
    img_path="",
    num_classes_aux=0,
    aux_mode="sigmoid",
):
    model = model.to(device).eval()

    dataset = NFLDatasetCls(
        df.copy(),
        transforms=get_transfos_cls(train=False),
        root=img_path,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    preds = np.empty(0)
    preds_aux = np.empty((0, num_classes_aux))
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            y_pred, y_pred_aux = model(images)

            y_pred = torch.sigmoid(y_pred).view(-1)
            preds = np.concatenate([preds, y_pred.detach().cpu().numpy()])

            if num_classes_aux:
                y_pred_aux = (
                    y_pred_aux.sigmoid()
                    if aux_mode == "sigmoid"
                    else y_pred_aux.softmax(-1)
                )
                preds_aux = np.concatenate([preds_aux, y_pred_aux.detach().cpu().numpy()])

    return preds, preds_aux
