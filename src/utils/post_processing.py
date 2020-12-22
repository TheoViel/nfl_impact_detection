import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.metrics import iou_score


def compute_ious(df, max_dist=10):
    ious = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(len(df)):
            frames = df["frame"].values[[i, j]]
            if np.abs(frames[0] - frames[1]) > max_dist:
                continue

            try:
                boxes = df[["left", "width", "top", "height"]].values[[i, j]]
            except KeyError:
                boxes = df[["x", "w", "y", "h"]].values[[i, j]]

            boxes[:, 1] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 2]
            boxes = boxes[:, [0, 2, 1, 3]]

            iou = iou_score(boxes[0], boxes[1])
            ious[i, j] = iou
            ious[j, i] = iou
    return ious


def form_clusters(df, threshold=0.5, max_dist=10):
    ious = compute_ious(df, max_dist=max_dist)

    frames = df["frame"]

    clust_mat = np.zeros((len(df), len(df)))

    for i in range(len(df)):
        for j in range(len(df)):
            if frames[i] == frames[j]:
                continue
            elif ious[i, j] > threshold:
                clust_mat[i, j] = 1
                clust_mat[j, i] = 1

    clusts = [[0]]
    for i in range(1, len(df)):
        in_clust = False
        for clust in clusts[::-1]:
            if clust_mat[clust[-1], i]:
                in_clust = True
                clust.append(i)
                break

        if not in_clust:
            clusts.append([i])

    centroids = [c[len(c) // 2] for c in clusts]

    return clusts, centroids


# def form_clusters(df, threshold=0.5, max_dist=10):
#     ious = compute_ious(df, max_dist=max_dist)

#     frames = df["frame"]

#     clust_mat = np.zeros((len(df), len(df)))

#     for i in range(len(df)):
#         for j in range(len(df)):
#             if frames[i] == frames[j]:
#                 continue
#             elif ious[i, j] > threshold:
#                 clust_mat[i, j] = ious[i, j]
#                 clust_mat[j, i] = ious[i, j]

#     clusts = [[0]]
#     for i in range(1, len(df)):
#         best_clust = None
#         best_iou = 0

#         for clust in clusts:
#             for j in clust:
#                 if clust_mat[j, i] > best_iou:
#                     best_clust = clust
#                     best_iou = best_iou

#         if best_clust is not None:
#             best_clust.append(i)
#         else:
#             clusts.append([i])

#     centroids = [c[len(c) // 2] for c in clusts]

#     return clusts, centroids


def post_process_adjacency(df, threshold=0.5, max_dist=10, min_clust_size=0):
    dfs_pp = []
    for video in tqdm(df["video"].unique()):
        df_video = df[df["video"] == video].reset_index(drop=True).copy()
        clusts, centroids = form_clusters(
            df_video, threshold=threshold, max_dist=max_dist
        )
        centroids = [
            centroids[i]
            for i in range(len(centroids))
            if len(clusts[i]) >= min_clust_size
        ]

        df_video_pp = (
            df_video.iloc[centroids].sort_values("frame").reset_index(drop=True)
        )
        dfs_pp.append(df_video_pp)

    df_pp = pd.concat(dfs_pp).reset_index(drop=True)
    return df_pp


def post_process_view(df, min_dist=4):
    to_drop = []
    for keys in tqdm(df.groupby(["gameKey", "playID"]).size().to_dict().keys()):

        tmp_df = df.query("gameKey == @keys[0] and playID == @keys[1]")
        tmp_to_drop = []
        for index, row in tmp_df.iterrows():
            if row["view"] == "Endzone":
                other_view = tmp_df.query('view == "Sideline"')
            else:
                other_view = tmp_df.query('view == "Endzone"')

            distances = other_view["frame"].apply(lambda x: np.abs(x - row["frame"]))
            if np.min(distances) > min_dist:
                tmp_to_drop.append(index)

        if len(tmp_to_drop) != len(tmp_df):
            to_drop += tmp_to_drop

    return df.drop(index=to_drop).reset_index(drop=True)


def post_process_view_2(df, min_dist=4):

    to_drop = []
    for keys in tqdm(df.groupby(["gameKey", "playID"]).size().to_dict().keys()):
        tmp_df = df.query("gameKey == @keys[0] and playID == @keys[1]")

        tmp_to_drop = []
        for index, row in tmp_df.iterrows():
            current_frame = row["frame"]  # noqa
            c_1 = tmp_df.query(
                'view == "Sideline" and abs(frame - @current_frame) <= 0'
            ).shape[0]
            c_2 = tmp_df.query(
                'view == "Endzone" and abs(frame - @current_frame) <= 0'
            ).shape[0]
            if c_1 != c_2:
                tmp_to_drop.append(index)

        if len(tmp_to_drop) != len(tmp_df):
            to_drop += tmp_to_drop

    return df.drop(index=to_drop).reset_index(drop=True)


def iou(box1, box2):
    """Compute the Intersection-Over-Union of two given boxes.
    Args:
      box1: array of 4 elements [cx, cy, width, height].
      box2: same as above
    Returns:
      iou: a float number in range [0, 1]. iou of the two boxes.
    """

    lr = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - max(
        box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2]
    )
    if lr > 0:
        tb = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - max(
            box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3]
        )
        if tb > 0:
            intersection = tb * lr
            union = box1[2] * box1[3] + box2[2] * box2[3] - intersection

            return intersection / union

    return 0


def batch_iou(boxes, box):
    """Compute the Intersection-Over-Union of a batch of boxes with another
    box.
    Args:
      box1: 2D array of [cx, cy, width, height].
      box2: a single array of [cx, cy, width, height]
    Returns:
      ious: array of a float number in range [0, 1].
    """
    lr = np.maximum(
        np.minimum(boxes[:, 0] + 0.5 * boxes[:, 2], box[0] + 0.5 * box[2])
        - np.maximum(boxes[:, 0] - 0.5 * boxes[:, 2], box[0] - 0.5 * box[2]),
        0,
    )
    tb = np.maximum(
        np.minimum(boxes[:, 1] + 0.5 * boxes[:, 3], box[1] + 0.5 * box[3])
        - np.maximum(boxes[:, 1] - 0.5 * boxes[:, 3], box[1] - 0.5 * box[3]),
        0,
    )
    inter = lr * tb
    index = np.where(abs(boxes[:, 4] - box[4]) > 6)
    inter[index] = 0
    union = boxes[:, 2] * boxes[:, 3] + box[2] * box[3] - inter
    return inter / union


def nms(boxes, probs, threshold):
    """Non-Maximum supression.
    Args:
      boxes: array of [cx, cy, w, h] (center format)
      probs: array of probabilities
      threshold: two boxes are considered overlapping if their IOU is largher than
          this threshold
      form: 'center' or 'diagonal'
    Returns:
      keep: array of True or False.
    """

    order = probs.argsort()[::-1]
    keep = [True] * len(order)
    for i in range(len(order) - 1):
        ovps = batch_iou(boxes[order[i + 1 :]], boxes[order[i]])
        for j, ov in enumerate(ovps):
            if ov > threshold:
                keep[order[j + i + 1]] = False
    return keep
