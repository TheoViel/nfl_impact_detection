# From https://www.kaggle.com/nvnnghia/evaluation-metrics

import numpy as np
from scipy.optimize import linear_sum_assignment


def get_boxes_from_df(df, videos):
    try:
        cols = ["frame", "video", "x", "w", "y", "h"]
        df = df[cols].groupby("video").agg(list)
    except KeyError:
        cols = ["frame", "video", "left", "width", "top", "height"]
        df = df[cols].groupby("video").agg(list)

    boxes = []

    for video in videos:
        try:
            frames, x, w, y, h = df.loc[video]
            boxes_pred = np.concatenate([
                np.array(frames)[:, None],
                np.array(x)[:, None],
                np.array(y)[:, None],
                np.array(w)[:, None] + np.array(x)[:, None],
                np.array(h)[:, None] + np.array(y)[:, None],
            ], -1)
            boxes.append(boxes_pred)
        except KeyError:
            boxes.append([])
    return boxes


def iou_score(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2

    # get the overlap rectangle
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)

    # check if there is an overlap
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0

    # if yes, calculate the ratio of the overlap to each ROI size and the unified size
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection

    return size_intersection / size_union


def precision_calc(gt_boxes, pred_boxes, return_assignment=False):
    cost_matix = np.ones((len(gt_boxes), len(pred_boxes)))
    for i, box1 in enumerate(gt_boxes):
        for j, box2 in enumerate(pred_boxes):
            dist = abs(box1[0] - box2[0])
            if dist > 4:
                continue
            iou = iou_score(box1[1:], box2[1:])

            if iou < 0.35:
                continue

            else:
                cost_matix[i, j] = 0

    row_ind, col_ind = linear_sum_assignment(cost_matix)

    if return_assignment:
        return cost_matix, row_ind, col_ind

    fn = len(gt_boxes) - row_ind.shape[0]
    fp = len(pred_boxes) - col_ind.shape[0]
    tp = 0
    for i, j in zip(row_ind, col_ind):
        if cost_matix[i, j] == 0:
            tp += 1
        else:
            fp += 1
            fn += 1

    return tp, fp, fn


def boxes_f1_score(preds, truths):
    """
    F1 score metric for the competition.

    Predictions and ground truths are lists of the predictions at video level.
    Each lists contains a list of all the boxes, represented as a 5 element tuple T :
        - T[0] : Frame
        - T[1:5]: boxe coordinate

    Args:
        preds (List): Predictions.
        truths (List): Truths.

    Returns:
        float: f1 score
    """
    ftp, ffp, ffn = [], [], []
    for pred, truth in zip(preds, truths):
        tp, fp, fn = precision_calc(truth, pred)
        ftp.append(tp)
        ffp.append(fp)
        ffn.append(fn)

    tp = np.sum(ftp)
    fp = np.sum(ffp)
    fn = np.sum(ffn)
    # print(tp, fp, fn)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    # print(precision, recall)

    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    return f1_score
