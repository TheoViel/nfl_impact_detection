import numpy as np


def expand_boxes(df, r=0):
    """
    Expands bounding boxes by a specified ratio.

    Args:
        df (pandas dataframe): Predictions of the format expected by the competition.
        r (int, optional): Ratio. Defaults to 0.

    Returns:
        pandas dataframe: Expanded predictions, of the format expected by the competition.
    """
    if r > 0:
        # Expansion
        df.left -= df.width * r / 2
        df.top -= df.height * r / 2
        df.width *= 1 + r
        df.height *= 1 + r
        df.left = np.clip(df.left, 0, None)
        df.top = np.clip(df.top, 0, None)
        df.width = np.clip(df.width, 0, 1280 - df.left)
        df.height = np.clip(df.height, 0, 720 - df.top)

        # Rounding

        right = np.round(df.left + df.width, 0)
        bot = np.round(df.top + df.height, 0)
        df.left = np.round(df.left, 0).astype(int)
        df.top = np.round(df.top, 0).astype(int)

        df.width = (right - df.left).astype(int)
        df.height = (bot - df.top).astype(int)

    return df
