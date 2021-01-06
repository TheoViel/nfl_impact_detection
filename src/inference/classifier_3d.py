import cv2
import torch
import numpy as np

from params import NUM_WORKERS
from utils.torch import load_model_weights
from torch.utils.data import DataLoader, Dataset

from model_zoo.models_cls_3d import get_model_cls_3d


def extend_box(box, size=64):
    w = box[1] - box[0]
    h = box[3] - box[2]

    dw = (size - w) / 2
    dh = (size - h) / 2

    new_box = [
        box[0] - np.floor(dw),
        box[1] + np.ceil(dw),
        box[2] - np.floor(dh),
        box[3] + np.ceil(dh),
    ]
    return np.array(new_box).astype(int)


def adapt_to_shape(box, shape):
    if box[0] < 0:
        box[1] -= box[0]
        box[0] = 0
    elif box[1] >= shape[1]:
        diff = box[1] - shape[1]
        box[1] -= diff
        box[0] -= diff

    if box[2] < 0:
        box[3] -= box[2]
        box[2] = 0

    elif box[3] >= shape[0]:
        diff = box[3] - shape[0]
        box[3] -= diff
        box[2] -= diff

    return box


def get_adjacent_frames(frame, max_frame=100, n_frames=9, stride=1):
    frames = np.arange(n_frames) * stride
    frames = frames - frames[n_frames // 2] + frame

    if frames.min() < 1:
        frames -= frames.min() - 1
    elif frames.max() > max_frame:
        frames += max_frame - frames.max()

    return frames


class NFLDatasetClsInference3D(Dataset):
    def __init__(self, df, n_frames=9, stride=2, visualize=False, root=""):
        super().__init__()
        self.n_frames = n_frames
        self.stride = stride
        self.visualize = visualize
        self.root = root

        self.max_frame = df['nb_frame'].values[0]

        self.images = []

        image_name = df['image_name'].values[0].split('.')[0][:-4]
        self.images = [image_name + f'{f:04d}.png' for f in range(1, self.max_frame + 1)]

        self.images = [cv2.imread(self.root + img) for img in self.images]

        self.frame_to_img = list(df["frame"].unique())
        self.frames = df["frame"].values

        self.boxes = df[["left", "width", "top", "height"]].values
        self.boxes[:, 1] += self.boxes[:, 0]
        self.boxes[:, 3] += self.boxes[:, 2]

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        frames = get_adjacent_frames(
            frame, max_frame=self.max_frame, n_frames=self.n_frames, stride=self.stride
        )

        image = [self.images[f - 1] for f in frames]
        image = np.array(image)

        box = extend_box(self.boxes[idx], size=64)
        box = adapt_to_shape(box, image.shape[1:])

        image = image[:, box[2]: box[3], box[0]: box[1]]

        if not self.visualize:
            image = (image / 255 - 0.5) / 0.5
            image = image.transpose(3, 0, 1, 2)
            image = torch.from_numpy(image).float()

        return image


def retrieve_model(config, fold=0, log_folder=""):
    model = get_model_cls_3d(
        config["name"],
        num_classes=config["num_classes"],
        num_classes_aux=config["num_classes_aux"],
        pretrained=False
    ).eval()
    model.zero_grad()

    model = load_model_weights(model, log_folder + f"{config['name']}_{fold}.pt")

    return model


def inference(df, models, batch_size=256, device="cuda", root="", n_frames=9, stride=2):
    models = [model.to(device).eval() for model in models]

    dataset = NFLDatasetClsInference3D(
        df.copy(),
        root=root,
        n_frames=n_frames,
        stride=stride,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    preds = []
    with torch.no_grad():
        for img in loader:
            img = img.to(device)
            preds_img = []

            for model in models:
                y_pred = model(img)[0]
                preds_img.append(y_pred.sigmoid().detach().cpu().numpy())

            preds.append(np.mean(preds_img, 0))

    return np.concatenate(preds)
