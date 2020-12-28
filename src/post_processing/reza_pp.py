import numpy as np
import pandas as pd


box_cols = ["left", "top", "width", "height"]


def iou_ltwh(bbox1, bbox2):

    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]

    bbox1[2] += bbox1[0]
    bbox1[3] += bbox1[1]

    bbox2[2] += bbox2[0]
    bbox2[3] += bbox2[1]

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


def camera_selection(test_df, min_dist=4, score_threshold=0.995):
    dropIDX = []
    for keys in test_df.groupby(["gameKey", "playID"]).size().to_dict().keys():
        tmp_df = test_df.query("gameKey == @keys[0] and playID == @keys[1]")

        for index, row in tmp_df.iterrows():
            if row["scores"] < score_threshold:
                if row["view"] == "Endzone":
                    check_df = tmp_df.query('view == "Sideline"')
                    if (
                        check_df["frame"]
                        .apply(lambda x: np.abs(x - row["frame"]) <= min_dist)
                        .sum()
                        == 0
                    ):
                        dropIDX.append(index)

                if row["view"] == "Sideline":
                    check_df = tmp_df.query('view == "Endzone"')
                    if (
                        check_df["frame"]
                        .apply(lambda x: np.abs(x - row["frame"]) <= min_dist)
                        .sum()
                        == 0
                    ):
                        dropIDX.append(index)

    return test_df.drop(index=dropIDX).reset_index(drop=True)


def post_process(baseline, threshold=0.4, max_dist=4):
    final = []
    for vid, rows in baseline.groupby("video"):
        queue = []
        for indx, row in rows.iterrows():
            frame = row["frame"]
            newbox = row[box_cols].tolist()
            added = False

            if not queue:
                queue.append([newbox, frame, indx, row["scores"]])
                added = True

            if not added:
                for item in queue:
                    # print(frame, item[1], iou_ltwh(newbox, item[0]))
                    if iou_ltwh(newbox, item[0]) > threshold and item[1] != frame:
                        added = True
                        if row["scores"] > item[3]:
                            item[2] = indx
                            item[3] = row["scores"]
                        item[0] = newbox
                        item[1] = frame

                        break
            if not added:
                queue.append([newbox, frame, indx, row["scores"]])

            for item in queue.copy():
                if frame - item[1] > max_dist:
                    final.append([vid] + item)
                    queue.remove(item)
        for item in queue.copy():
            final.append([vid] + item)
            queue.remove(item)

    return baseline.loc[pd.DataFrame(final)[3].tolist(), :].copy()
