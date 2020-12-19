import numpy as np
import pandas as pd

from tqdm import tqdm
from utils.metrics import iou_score


def compute_ious(df, max_dist=10):
    ious = np.zeros((len(df), len(df)))
    for i in range(len(df)):
        for j in range(len(df)):
            frames = df['frame'].values[[i, j]]
            if np.abs(frames[0] - frames[1]) > max_dist:
                continue

            try:
                boxes = df[['left', 'width', 'top', 'height']].values[[i, j]]
            except KeyError:
                boxes = df[['x', 'w', 'y', 'h']].values[[i, j]]

            boxes[:, 1] += boxes[:, 0]
            boxes[:, 3] += boxes[:, 2]
            boxes = boxes[:, [0, 2, 1, 3]]

            iou = iou_score(boxes[0], boxes[1])
            ious[i, j] = iou
            ious[j, i] = iou
    return ious


def form_clusters(df, threshold=0.5, max_dist=10):
    ious = compute_ious(df, max_dist=max_dist)

    frames = df['frame']

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
        for clust in clusts:
            if clust_mat[clust[-1], i]:
                in_clust = True
                clust.append(i)

        if not in_clust:
            clusts.append([i])

    centroids = [c[len(c) // 2] for c in clusts]
    # centroids = [c[0] for c in clusts]

    return clusts, centroids


def post_process_adjacency(df, threshold=0.5, max_dist=10, min_clust_size=0):
    dfs_pp = []
    for video in tqdm(df['video'].unique()):
        df_video = df[df['video'] == video].reset_index(drop=True).copy()
        clusts, centroids = form_clusters(df_video, threshold=threshold, max_dist=max_dist)

        centroids = [
            centroids[i] for i in range(len(centroids)) if len(clusts[i]) >= min_clust_size
        ]

        df_video_pp = df_video.iloc[centroids].sort_values('frame').reset_index(drop=True)
        dfs_pp.append(df_video_pp)

    df_pp = pd.concat(dfs_pp).reset_index(drop=True)
    return df_pp


def post_process_view(df, min_dist=4):
    to_drop = []
    for keys in tqdm(df.groupby(['gameKey', 'playID']).size().to_dict().keys()):

        tmp_df = df.query('gameKey == @keys[0] and playID == @keys[1]')
        tmp_to_drop = []
        for index, row in tmp_df.iterrows():
            if row['view'] == 'Endzone':
                other_view = tmp_df.query('view == "Sideline"')
            else:
                other_view = tmp_df.query('view == "Endzone"')

            distances = other_view['frame'].apply(lambda x: np.abs(x - row['frame']))
            if np.min(distances) > min_dist:
                tmp_to_drop.append(index)

        if len(tmp_to_drop) != len(tmp_df):
            to_drop += tmp_to_drop

    return df.drop(index=to_drop).reset_index(drop=True)


def post_process_view_2(df, min_dist=4):

    to_drop = []
    for keys in tqdm(df.groupby(['gameKey', 'playID']).size().to_dict().keys()):
        tmp_df = df.query('gameKey == @keys[0] and playID == @keys[1]')

        tmp_to_drop = []
        for index, row in tmp_df.iterrows():
            current_frame = row['frame']  # noqa
            c_1 = tmp_df.query('view == "Sideline" and abs(frame - @current_frame) <= 0').shape[0]
            c_2 = tmp_df.query('view == "Endzone" and abs(frame - @current_frame) <= 0').shape[0]
            if c_1 != c_2:
                tmp_to_drop.append(index)

        if len(tmp_to_drop) != len(tmp_df):
            to_drop += tmp_to_drop

    return df.drop(index=to_drop).reset_index(drop=True)
