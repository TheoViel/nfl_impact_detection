import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

cv2.setNumThreads(0)


class NFLDatasetCls3D(Dataset):
    """
    Torch Dataset for the problem
    """
    def __init__(self, df, target_name="impact", root="", visualize=False):
        """
        Constructor

        Args:
            df (pandas dataframe): Data.
            target_name (str, optional): Name of the target. Defaults to "impact".
            root (str, optional): Directory containing the data. Defaults to "".
            visualize (bool, optional): Whether to return an image for plotting. Defaults to False.
        """
        super().__init__()
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

        if not self.visualize:
            image = (image / 255 - 0.5) / 0.5
            image = image.transpose(3, 0, 1, 2)

            image = torch.from_numpy(image).float()

        return image, self.labels[idx], self.aux_labels[idx]
