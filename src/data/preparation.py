import cv2
import numpy as np
import matplotlib.pyplot as plt


def extend_box(box, size=64):
    """
    Extends a bounding box to be of a chosen size.

    Args:
        box (numpy array ): Bounding box.
        size (int, optional): Target size. Defaults to 64.

    Returns:
        numpy array: Extended bounding box.
    """
    w = box[1] - box[0]
    h = box[3] - box[2]

    dw = (size - w) / 2
    dh = (size - h) / 2

    new_box = [
        box[0] - np.floor(dw),
        box[1] + np.ceil(dw),
        box[2] - np.floor(dh),
        box[3] + np.ceil(dh),
    ]
    return np.array(new_box).astype(int)


def adapt_to_shape(box, shape):
    """
    Modifies a bounding box to fit in a given shape.

    Args:
        box (numpy array): Bounding box.
        shape (numpy array): Shape (H, W).

    Returns:
        numpy array: Adapted bounding box.
    """
    if box[0] < 0:
        box[1] -= box[0]
        box[0] = 0
    elif box[1] >= shape[1]:
        diff = box[1] - shape[1]
        box[1] -= diff
        box[0] -= diff

    if box[2] < 0:
        box[3] -= box[2]
        box[2] = 0

    elif box[3] >= shape[0]:
        diff = box[3] - shape[0]
        box[3] -= diff
        box[2] -= diff

    return box


def load_adjacent_images(image_name, max_frame=100, n_frames=9, stride=1, root=""):
    """
    Loads an image and its neighbours.

    Args:
        image_name ([type]): Name of the image.
        max_frame (int, optional): Maximum frame available. Defaults to 100.
        n_frames (int, optional): Number of frames. Defaults to 9.
        stride (int, optional): Spacing between frames. Defaults to 1.
        root (str, optional): Images directory. Defaults to "".

    Returns:
        numpy array [N_FRAMES x H x W x C]: Images.
    """
    if ".png" in image_name:
        image_name = image_name.split(".")[0]
        image_name, frame = image_name[:-5], int(image_name[-4:])

    frames = np.arange(n_frames) * stride
    frames = frames - frames[n_frames // 2] + frame

    if frames.min() < 1:
        frames -= frames.min() - 1
    elif frames.max() > max_frame:
        frames += max_frame - frames.max()

    img_names = [image_name + f"_{f:04d}.png" for f in frames]
    images = np.array([cv2.imread(root + img) for img in img_names])

    return np.array(images)


def crop_helmets(image_name, df, size=64, n_frames=9, stride=2, root="", out_dir="", plot=False):
    """
    Computes 3D crops around helmet boxes to feed to the network.

    Args:
        image_name (str): Image name.
        df (pandas dataframe): Metadata
        size (int, optional): Size of the crop. Defaults to 64.
        n_frames (int, optional): Number of frames to use. Defaults to 9.
        stride (int, optional): Spacing between frames. Defaults to 2.
        root (str, optional): Path to load image from. Defaults to "".
        out_dir (str, optional): Path to save crops to. Defaults to "".
        plot (bool, optional): Whether to plot crops. Defaults to False.

    Returns:
        list of str: Names of the generated crops.
    """
    df_img = df[df["image_name"] == image_name].copy()

    images = load_adjacent_images(
        image_name,
        max_frame=df_img["nb_frame"].values[0],
        n_frames=n_frames,
        stride=stride,
        root=root,
    )

    boxes = df_img[["x", "w", "y", "h"]].values
    boxes[:, 1] += boxes[:, 0]
    boxes[:, 3] += boxes[:, 2]

    crop_names = []
    for i in range(len(df_img)):
        box = boxes[i]
        box = extend_box(box, size=size)

        img_crop = images[:, box[2]: box[3], box[0]: box[1]]

        name = image_name[:-4] + f"_{i:02d}.npy"

        if df_img["impact"].values[i] and np.random.random() < 0.1:
            plt.figure(figsize=(15, 15))
            print(name)
            for i, img in enumerate(img_crop):
                plt.subplot(5, 5, i + 1)
                plt.imshow(img)
                plt.axis(False)
            plt.show()

        try:
            if img_crop.shape == (n_frames, 64, 64, 3):
                np.save(out_dir + name, img_crop)
            else:
                name = ""
        except:  # noqa
            name = ""

        crop_names.append(name)

    return crop_names
