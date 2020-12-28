import numpy as np
import pandas as pd

from tqdm.notebook import tqdm

from post_processing.expansion import expand_boxes


def frame_box_dist(b1, b2):
    def get_center(box):
        return (box[0] + box[1] / 2, box[2] + box[3] / 2)

    frame_dist = np.abs(b1[0] - b2[0])

    x1, y1 = get_center(b1[1:])
    x2, y2 = get_center(b2[1:])

    box_dist = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    return frame_dist, box_dist


def pair_impacts(df, alpha=10, max_frame_dist=4, max_box_dist=10, pair_ground=False):
    if "predicted_impact_type" in df.columns and pair_ground:  # exclude ground impacts
        df_h = df[df["predicted_impact_type"] != "ground"]
        df_ground = df[df["predicted_impact_type"] == "ground"]
    else:
        df_h = df

    try:
        boxes = df_h[["frame", "x", "w", "y", "h"]].values
    except KeyError:
        boxes = df_h[["frame", "left", "width", "top", "height"]].values

    # Compute distances
    distances = np.zeros((len(boxes), len(boxes)))
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            frame_dist, box_dist = frame_box_dist(boxes[i], boxes[j])
            dist = box_dist + alpha * frame_dist
            if frame_dist < max_frame_dist and box_dist < max_box_dist:
                distances[i, j] = -1 / (dist + 1)
                distances[j, i] = -1 / (dist + 1)

    # Find pairings
    paired = []
    pairs = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if np.max(-distances[j]) and np.max(-distances[i]):
                if i == np.argmin(distances[j]) and j == np.argmin(distances[i]):
                    paired += [i, j]
                    pairs.append((i, j))

    # Paired impacts
    unpaired = []
    for i, (index, _) in enumerate(df_h.iterrows()):
        if i not in paired:
            unpaired.append(index)

    # Ground impacts are paired
    if "predicted_impact_type" in df.columns and pair_ground:
        for i, (index, _) in enumerate(df_ground.iterrows()):
            paired.append(index)

    return paired, unpaired


def retrieve_closest(
    player, candidates_df, max_dist=100, verbose=0, cls_threshold=0, det_threshold=0
):
    candidates_df = candidates_df[candidates_df["frame"] == player["frame"].values[0]]
    candidates_df = candidates_df[candidates_df["pred_cls"] > cls_threshold]
    candidates_df = candidates_df[candidates_df["pred"] > det_threshold].reset_index(
        drop=True
    )

    try:
        box = player[["frame", "x", "w", "y", "h"]].values[0]
        candidates = candidates_df[["frame", "x", "w", "y", "h"]].values
    except KeyError:
        box = player[["frame", "left", "width", "top", "height"]].values[0]
        candidates = candidates_df[["frame", "left", "width", "top", "height"]].values

    # Compute distances
    distances = []
    for i in range(len(candidates)):
        distances.append(frame_box_dist(box, candidates[i])[1])

    # Retrieve closest, if it's closer than max_dist
    if len(distances) > 1:
        selected = np.argsort(distances)[1]
        if distances[selected] < max_dist:
            if verbose:
                print(f'Found candidate at frame {player["frame"].values[0]}')
            return candidates_df.iloc[[selected]]
    if verbose:
        print(f'No candidate found at frame {player["frame"].values[0]}')
    return None


def post_process_pairing(
    df,
    candidates,
    alpha=10,
    max_box_dist=100,
    max_frame_dist=8,
    r=0,
    max_dist=100,
    cls_threshold=0,
    det_threshold=0,
    verbose=0,
    remove_unpaired=True,
):
    new_impacts = []
    df_paired = []

    for video in tqdm(df["video"].unique()):
        tmp_candidates = candidates.query("video == @video").reset_index(drop=True)
        tmp_df = df.query("video == @video").reset_index(drop=True)

        paired, unpaired = pair_impacts(
            tmp_df,
            alpha=alpha,
            max_box_dist=max_box_dist,
            max_frame_dist=max_frame_dist,
        )

        if verbose:
            print(f"Found {len(unpaired)} unpaired indices.")

        for idx in unpaired:
            closest = retrieve_closest(
                tmp_df.iloc[[idx]],
                tmp_candidates,
                max_dist=max_dist,
                cls_threshold=cls_threshold,
                det_threshold=det_threshold,
                verbose=verbose,
            )
            if closest is not None:
                new_impacts.append(closest)
                paired.append(idx)

        if remove_unpaired:
            df_paired.append(tmp_df.iloc[paired])
        else:
            df_paired.append(tmp_df)

    new_impacts = pd.concat(new_impacts).reset_index(drop=True)
    new_impacts = expand_boxes(new_impacts, r=r)

    df_paired = pd.concat(df_paired).reset_index(drop=True)

    return (
        pd.concat([df_paired, new_impacts])
        .sort_values(["gameKey", "playID", "view", "frame"])
        .reset_index(drop=True)
    )
