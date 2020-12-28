import numpy as np
from tqdm.notebook import tqdm


def post_process_view(df, min_dist=4, keep_ground=False):
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
                if not keep_ground:
                    tmp_to_drop.append(index)
                else:
                    if row["predicted_impact_type"] != "ground":
                        tmp_to_drop.append(index)
                    else:
                        if row["view"] == "Endzone":
                            tmp_to_drop.append(index)
                        else:
                            print('Keep ground')

        if len(tmp_to_drop) != len(tmp_df):
            to_drop += tmp_to_drop

    return df.drop(index=to_drop).reset_index(drop=True)
