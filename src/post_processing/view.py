import numpy as np


def post_process_view(df, min_dist=4, threshold=1., cls_col="pred_cls"):
    """
    Post-processing to remove impacts not present in both views.
    An impact is removed if the two following conditions are met :
      - It is predicted with a confidence < threshold
      - No other impact is found in the other view within +/- min_dist frames

    Args:
        df (pandas dataframe): Predictions of the format expected by the competition.
        min_dist (int, optional): Margin for associating impacts between views. Defaults to 4.
        threshold ([type], optional): Confidence threshold for removal. Defaults to 1..
        cls_col (str, optional): Column to consider for confidences. Defaults to "pred_cls".

    Returns:
        pandas dataframe: Filtered predictions, of the format expected by the competition.
    """
    to_drop = []
    for keys in df.groupby(["gameKey", "playID"]).size().to_dict().keys():

        tmp_df = df.query("gameKey == @keys[0] and playID == @keys[1]")
        tmp_to_drop = []
        for index, row in tmp_df.iterrows():
            if row["view"] == "Endzone":
                other_view = tmp_df.query('view == "Sideline"')
            else:
                other_view = tmp_df.query('view == "Endzone"')

            distances = other_view["frame"].apply(lambda x: np.abs(x - row["frame"]))
            if (
                (np.min(distances) > min_dist)
                & (row['pred'] < threshold)
                & (row[cls_col] < threshold)
            ):
                tmp_to_drop.append(index)

        if len(tmp_to_drop) != len(tmp_df):
            to_drop += tmp_to_drop

    return df.drop(index=to_drop).reset_index(drop=True)
