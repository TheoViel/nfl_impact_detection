import cv2
import numpy as np
import matplotlib.pyplot as plt

from itertools import product
from sklearn.metrics import confusion_matrix

COLOR_DIC = {
    'r': ((255, 0, 0), (1, 0, 0)),
    'g': ((0, 255, 0), (0, 1, 0)),
    'b': ((0, 0, 255), (0, 0, 1)),
    'orange': ((255, 165, 0), (1, 0.65, 0)),
}


def plot_bboxes(img, boxes, label, transpose=False):
    thickness = 2 if img.shape[1] > 400 else 1
    for box, label in zip(boxes, label):
        if img.max() > 1:
            color = (255, 0, 0) if label == 2 else (0, 255, 0)
        else:
            color = (1, 0, 0) if label == 2 else (0, 1, 0)

        if not transpose:
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]), color, thickness=thickness
            )
        else:
            cv2.rectangle(
                img, (box[1], box[0]), (box[3], box[2]), color, thickness=thickness
            )

    plt.imshow(img)


def plot_bboxes_indexed(img, boxes, transpose=False, start_idx=0):
    thickness = 2 if img.shape[1] > 400 else 1
    for i, box in enumerate(boxes):
        if img.max() > 1:
            color = (255, 0, 0)
        else:
            color = (1, 0, 0)

        if not transpose:
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]), color, thickness=thickness
            )
        else:
            cv2.rectangle(
                img, (box[1], box[0]), (box[3], box[2]), color, thickness=thickness
            )
        plt.text(box[0], box[1] - 5, f'#{i + start_idx}', c='r', size=9)

    plt.imshow(img)


def plot_bboxes_pred(img, boxes, labels, colors, transpose=False):
    thickness = 2 if img.shape[1] > 400 else 1
    for box, label, c in zip(boxes, labels, colors):
        color = COLOR_DIC[c][int(img.max() <= 1)]

        if not transpose:
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]), color, thickness=thickness
            )
        else:
            cv2.rectangle(
                img, (box[1], box[0]), (box[3], box[2]), color, thickness=thickness
            )

        plt.text(box[0], box[1] - 5, label, c=c, size=12)

    plt.imshow(img)


class ConfusionMatrixDisplay:
    """
    Adapted from sklearn :
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
    """

    def __init__(self, confusion_matrix, display_labels):
        self.confusion_matrix = confusion_matrix
        self.display_labels = display_labels

    def plot(self, cmap="viridis", figsize=(10, 10), normalize=None):
        fig, ax = plt.subplots(figsize=figsize)

        # Display colormap
        cm = self.confusion_matrix
        n_classes = cm.shape[0]
        self.im_ = ax.imshow(cm, interpolation="nearest", cmap=cmap)

        # Display values
        self.text_ = np.empty_like(cm, dtype=object)
        cmap_min, cmap_max = self.im_.cmap(0), self.im_.cmap(256)
        thresh = (cm.max() + cm.min()) / 2.0
        for i, j in product(range(n_classes), range(n_classes)):
            color = cmap_max if cm[i, j] < thresh else cmap_min
            text = f"{cm[i, j]:.0f}" if normalize is None else f"{cm[i, j]:.3f}"
            self.text_[i, j] = ax.text(
                j, i, text, ha="center", va="center", color=color
            )

        # Display legend
        ax.set(
            xticks=np.arange(n_classes),
            yticks=np.arange(n_classes),
            xticklabels=self.display_labels,
            yticklabels=self.display_labels,
        )

        ax.set_ylabel("True label", fontsize=12)
        ax.set_xlabel("Predicted label", fontsize=12)

        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.tick_params(axis="both", which="minor", labelsize=11)

        ax.set_ylim((n_classes - 0.5, -0.5))
        plt.setp(ax.get_xticklabels(), rotation=40)

        self.figure_ = fig
        self.ax_ = ax
        return self


def plot_confusion_matrix(
    y_pred,
    y_true,
    normalize=None,
    display_labels=None,
    cmap="viridis",
    figsize=(10, 10),
):
    """
    Computes and plots a confusion matrix.

    Args:
        y_pred (numpy array): Predictions.
        y_true (numpy array): Truths.
        normalize (bool or None, optional): Whether to normalize the matrix. Defaults to None.
        display_labels (list of strings or None, optional): Axis labels. Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "viridis".
        figsize (tuple of 2 ints, optional): Figure size. Defaults to (10, 10).

    Returns:
        ConfusionMatrixDisplay: Confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
    return disp.plot(cmap=cmap, figsize=figsize, normalize=normalize)


def visualize_preds(df_pred, video_name, frame, root="", truth_col="impact", threshold_pred=0.7):
    img = f"{video_name[:-4]}_{frame:04d}.png"
    img = cv2.imread(root + img)

    df = df_pred[df_pred["video"] == video_name]
    df = df[df["frame"] == frame].reset_index(drop=True)

    try:
        boxes = df[["left", "width", "top", "height"]].values
    except KeyError:
        boxes = df[["x", "w", "y", "h"]].values

    boxes[:, 1] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 2]
    boxes = boxes[:, [0, 2, 1, 3]]

    try:
        labels = [f"{l}\n{s:.3f}" for s, l in df[["pred", "predicted_impact_type"]].values]
    except KeyError:
        labels = [f"{s:.3f}" for s in df["pred"].values]

    colors = []
    if "match" in df.columns:
        for pred, truth, match in df[["pred", truth_col, "match"]].values:
            if truth:
                if match:
                    colors.append("g")
                else:
                    colors.append("r")
            elif pred > threshold_pred:
                if match:
                    colors.append("b")
                else:
                    colors.append("orange")
    else:
        for pred, truth in df[["pred", truth_col]].values:
            if truth:
                if pred > threshold_pred:
                    colors.append("g")
                else:
                    colors.append("r")
            else:
                colors.append("b")

    plot_bboxes_pred(img, boxes, labels, colors)
    plt.title(f"Video {video_name} - frame {frame}")


def visualize_preds_indexed(df_pred, idx, root='', start_idx=0):
    video_name = df_pred['video'][idx]
    frame = df_pred['frame'][idx]
    img = f"{video_name[:-4]}_{frame:04d}.png"
    img = cv2.imread(root + img)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    df = df_pred[df_pred["video"] == video_name]
    df = df[df['frame'] == frame]

    boxes = df[['left', 'width', 'top', 'height']].values
    boxes[:, 1] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 2]
    boxes = boxes[:, [0, 2, 1, 3]]

    plot_bboxes_indexed(img, boxes, start_idx=start_idx)
    plt.title(f'Video {video_name} - frame {frame}')
