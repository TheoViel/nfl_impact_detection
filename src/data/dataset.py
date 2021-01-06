import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

cv2.setNumThreads(0)


class NFLDatasetDet(Dataset):
    def __init__(self, df, transforms=None, root="", train=False):
        super().__init__()
        self.train = train
        self.transforms = transforms
        self.root = root

        self.images = df.image_name.unique()
        self.labels = []
        self.boxes = []

        df = df.copy()
        df['w'] += df['x']
        df['h'] += df['y']
        group = (
            df[["image_name", "x", "y", "w", "h", "impact", "extended_impact"]]
            .groupby("image_name")
            .agg(list)
            .reset_index()
        )
        self.images = group["image_name"].values

        if self.train:
            self.labels = group['extended_impact'].values.tolist()
        else:
            self.labels = group['impact'].values.tolist()

        self.boxes = group[['x', 'y', 'w', 'h']].values
        self.boxes = [np.array(box.tolist()).T for box in self.boxes]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(f"{self.root}/{self.images[idx]}", cv2.IMREAD_COLOR)

        boxes = self.boxes[idx]
        labels = self.labels[idx]

        frame = int(self.images[idx][-8:-4])
        vid = self.images[idx][:-9]

        if self.transforms:
            for i in range(100):
                sample = self.transforms(
                    **{"image": image, "bboxes": boxes, "labels": labels}
                )
                if len(sample["bboxes"]) > 0:  # make sure we still have bboxes
                    image = sample["image"]
                    labels = sample["labels"]

                    boxes = torch.stack(
                        tuple(map(torch.tensor, zip(*sample["bboxes"])))
                    ).permute(1, 0)

                    boxes[:, [0, 1, 2, 3]] = boxes[:, [1, 0, 3, 2]]  # (x, y), -> (y, x)
                    break

        return image, boxes.float(), torch.tensor(labels), vid, frame


class NFLDatasetCls(Dataset):
    def __init__(self, df, transforms=None, target_name="impact", root=""):
        super().__init__()
        self.transforms = transforms
        self.root = root
        self.images = df["crop_name"].values
        self.labels = df[target_name].values
        self.aux_labels = np.array(list(df['aux_target'].values))

        self.players = (df['video'] + "_" + df['label']).values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(f"{self.root}/{self.images[idx]}")

        if self.transforms:
            image = self.transforms(image=image)["image"]

        return image, self.labels[idx], self.aux_labels[idx]


class NFLDatasetCls3D(Dataset):
    def __init__(self, df, transforms=None, target_name="impact", root="", visualize=False):
        super().__init__()
        self.transforms = transforms
        self.visualize = visualize
        self.root = root
        self.images = df["crop_name"].values
        self.labels = df[target_name].values
        self.aux_labels = np.array(list(df['aux_target'].values))

        self.players = (df['video'] + "_" + df['label']).values

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(f"{self.root}/{self.images[idx]}")

        if self.transforms:
            image = self.transforms(image=image)["image"]

        if not self.visualize:
            image = (image / 255 - 0.5) / 0.5
            image = image.transpose(3, 0, 1, 2)

            image = torch.from_numpy(image).float()

        return image, self.labels[idx], self.aux_labels[idx]
