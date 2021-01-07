# From https://github.com/open-mmlab/mmaction2

import torch


CONFIGS = {
    "slowfast": dict(
        type="Recognizer3D",
        backbone=dict(
            type="ResNet3dSlowFast",
            pretrained=None,
            resample_rate=8,  # tau
            speed_ratio=8,  # alpha
            channel_ratio=8,  # beta_inv
            slow_pathway=dict(
                type="resnet3d",
                depth=50,
                pretrained=None,
                lateral=True,
                conv1_kernel=(1, 7, 7),
                dilations=(1, 1, 1, 1),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1),
                norm_eval=False,
            ),
            fast_pathway=dict(
                type="resnet3d",
                depth=50,
                pretrained=None,
                lateral=False,
                base_channels=8,
                conv1_kernel=(5, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                norm_eval=False,
            ),
        ),
        cls_head=dict(
            type="SlowFastHead",
            in_channels=2304,  # 2048+256
            num_classes=400,
            spatial_type="avg",
            dropout_ratio=0.5,
        ),
    ),
    "slowonly": dict(
        type="Recognizer3D",
        backbone=dict(
            type="ResNet3dSlowOnly",
            depth=50,
            pretrained=None,
            lateral=False,
            conv1_kernel=(1, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False,
        ),
        cls_head=dict(
            type="I3DHead",
            in_channels=2048,
            num_classes=400,
            spatial_type="avg",
            dropout_ratio=0.5,
        ),
    ),
}


def forward_slowfast(self, x):
    x = x[:, :, 1:, :, :]  # size BS x 8 x C x W x H
    ft1, ft2 = self.extract_feat(x)

    ft1 = self.avg_pool(ft1).view(x.size(0), -1)
    ft2 = self.avg_pool(ft2).view(x.size(0), -1)

    ft = torch.cat([ft1, ft2], -1)
    ft = self.dropout(ft)

    y = self.fc(ft)

    if self.num_classes_aux > 0:
        y_aux = self.fc_aux(ft)
        return y, y_aux

    return y, 0


def forward_slowonly(self, x):
    ft = self.extract_feat(x)

    ft = self.avg_pool(ft).view(x.size(0), -1)
    ft = self.dropout(ft)

    y = self.fc(ft)

    if self.num_classes_aux > 0:
        y_aux = self.fc_aux(ft)
        return y, y_aux

    return y, 0
