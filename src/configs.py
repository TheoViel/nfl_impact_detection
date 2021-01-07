# Training configs of the final models.

import torch
from params import PATH, STRIDE, N_FRAMES

BATCH_SIZES = {
    "i3d": 32,
    "slowfast": 64,
    "slowonly": 32,
    "resnet18": 128,
    "resnet34": 64,
    "resnet50": 32,
}


class ConfigI3d:
    """
    Inception-3D with default set-up.
    """

    # General
    seed = 42
    verbose = 1
    img_path = PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Target
    target_name = "extended_impact"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "i3d"
    num_classes = 1

    aux_mode = "softmax"
    num_classes_aux = 0

    # Training
    batch_size = BATCH_SIZES[name]
    samples_per_player = 4
    optimizer = "Adam"

    acc_steps = 1
    epochs = 20 if samples_per_player else 4
    swa_first_epoch = 15

    lr = 5e-4
    warmup_prop = 0.05
    val_bs = batch_size * 2

    first_epoch_eval = 0


class ConfigSlowFast:
    """
    Slowfast with default set-up.
    """

    # General
    seed = 42
    verbose = 1
    img_path = PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Target
    target_name = "extended_impact"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "slowfast"
    num_classes = 1

    aux_mode = "softmax"
    num_classes_aux = 0

    # Training
    batch_size = BATCH_SIZES[name]
    samples_per_player = 4
    optimizer = "Adam"

    acc_steps = 1
    epochs = 20 if samples_per_player else 4
    swa_first_epoch = 15

    lr = 5e-4
    warmup_prop = 0.05
    val_bs = batch_size * 2

    first_epoch_eval = 0


class ConfigSlowOnly:
    """
    Omnisource pretrained SlowOnly with default set-up.
    """

    # General
    seed = 42
    verbose = 1
    img_path = PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Target
    target_name = "extended_impact"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "slowonly"
    num_classes = 1

    aux_mode = "softmax"
    num_classes_aux = 0

    # Training
    batch_size = BATCH_SIZES[name]
    samples_per_player = 4
    optimizer = "Adam"

    acc_steps = 1
    epochs = 20 if samples_per_player else 4
    swa_first_epoch = 15

    lr = 5e-4
    warmup_prop = 0.05
    val_bs = batch_size * 2

    first_epoch_eval = 0


class ConfigResnet34:
    """
    ResNet-34 with default set-up.
    """

    # General
    seed = 42
    verbose = 1
    img_path = PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Target
    target_name = "extended_impact"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "resnet34"
    num_classes = 1

    aux_mode = "softmax"
    num_classes_aux = 0

    # Training
    batch_size = BATCH_SIZES[name]
    samples_per_player = 4
    optimizer = "Adam"

    acc_steps = 1
    epochs = 20 if samples_per_player else 4
    swa_first_epoch = 15

    lr = 5e-4
    warmup_prop = 0.05
    val_bs = batch_size * 2

    first_epoch_eval = 0


class ConfigResnet18:
    """
    ResNet-18 with default set-up.
    """

    # General
    seed = 42
    verbose = 1
    img_path = PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Target
    target_name = "extended_impact"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "resnet18"
    num_classes = 1

    aux_mode = "softmax"
    num_classes_aux = 0

    # Training
    batch_size = BATCH_SIZES[name]
    samples_per_player = 4
    optimizer = "Adam"

    acc_steps = 1
    epochs = 20 if samples_per_player else 4
    swa_first_epoch = 15

    lr = 5e-4
    warmup_prop = 0.05
    val_bs = batch_size * 2

    first_epoch_eval = 0


class ConfigResnet18Aux:
    """
    ResNet-18 with auxiliary classifier.
    """

    # General
    seed = 42
    verbose = 1
    img_path = PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Target
    target_name = "extended_impact"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "resnet18"
    num_classes = 1

    aux_mode = "softmax"
    num_classes_aux = 4

    # Training
    batch_size = BATCH_SIZES[name]
    samples_per_player = 4
    optimizer = "Adam"

    acc_steps = 1
    epochs = 20 if samples_per_player else 4
    swa_first_epoch = 15

    lr = 5e-4
    warmup_prop = 0.05
    val_bs = batch_size * 2

    first_epoch_eval = 0


class ConfigResnet18Ext:
    """
    ResNet-18 with extended target.
    """

    # General
    seed = 42
    verbose = 1
    img_path = PATH
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_weights = True

    # Target
    target_name = f"impact_{STRIDE}_{N_FRAMES}"

    # k-fold
    k = 5
    random_state = 0
    selected_folds = [0, 1, 2, 3, 4]

    # Model
    name = "resnet18"
    num_classes = 1

    aux_mode = "softmax"
    num_classes_aux = 0

    # Training
    batch_size = BATCH_SIZES[name]
    samples_per_player = 4
    optimizer = "Adam"

    acc_steps = 1
    epochs = 20 if samples_per_player else 4
    swa_first_epoch = 15

    lr = 5e-4
    warmup_prop = 0.05
    val_bs = batch_size * 2

    first_epoch_eval = 0
