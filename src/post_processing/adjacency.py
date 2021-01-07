import numpy as np
import pandas as pd

from tqdm.notebook import tqdm
from utils.metrics import iou_score


def compute_ious(df, max_dist=10):
    """
    Computes ious between boxes. If boxes are too far the iou is set to 0.

    Args:
        df (pandas dataframe): Predicted boxes.
        max_dist (int, optional): Maximum frame distance to compute iou for. Defaults to 10.

    Returns:
        np array [len(df) x len(df)]: ious between boxes.
    """
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


def get_centroids(clusts):
    """
    Returns the middle of a cluster.

    Args:
        clusts (list of lists of ints): Clusters.

    Returns:
        list of ints: Centroids.
    """
    centroids = []
    for clust in clusts:
        if len(clust) == 1:
            centroids.append(clust[0])
        elif (len(clust) % 2) == 1:
            centroids.append(clust[len(clust) // 2 + 1])
        else:
            centroids.append(clust[len(clust) // 2])

    return centroids


def form_clusters(df, threshold=0.5, max_dist=10):
    """
    Forms cluster identified as being from the same impact.
    The algorithm loops over the boxes, in frame order.
    Each box is assigned to the cluster of the box has the one it has the highest IOU with.
    Only IOUs > threshold are considered, and if no box is found, it is given a new cluster.
    In the end, only the centroid of the cluster is kept.

    Args:
        df (pandas dataframe): Predicted boxes.
        threshold (float, optional): Threshold for iou. Defaults to 0.5.
        max_dist (int, optional): Maximum frame distance. Defaults to 10.

    Returns:
        list of lists of ints: Clusters.
        list of ints : Associad centroids.
    """
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

    centroids = get_centroids(clusts)

    return clusts, centroids


def post_process_adjacency(df, threshold=0.5, max_dist=10, min_clust_size=0):
    """
    Post-processing to make sure impacts are only counted once.
    For all videos, the algorithm loops over the boxes, in frame order.
    Each box is assigned to the cluster of the box has the one it has the highest IOU with.
    Only IOUs > threshold are considered, and if no box is found, it is given a new cluster.
    In the end, only the centroid of the cluster is kept.

    Args:
        df (pandas dataframe): Predictions of the format expected by the competition.
        threshold (float, optional): Threshold for iou. Defaults to 0.5.
        max_dist (int, optional): Maximum frame distance. Defaults to 10.
        min_clust_size (int, optional): Minimum cluster size to consider. Defaults to 0.

    Returns:
        pandas dataframe: Processed predictions, of the format expected by the competition.
    """
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
