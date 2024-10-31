# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))
model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='ViTAE_Window_NoShift_basic',
        RC_tokens_type=['swin', 'swin', 'transformer', 'transformer'],
        NC_tokens_type=['swin', 'swin', 'transformer', 'transformer'],
        stages=4,
        embed_dims=[64, 64, 128, 256],
        token_dims=[64, 128, 256, 512],
        downsample_ratios=[4, 2, 2, 2],
        NC_depth=[2, 2, 8, 2],
        NC_heads=[1, 2, 4, 8],
        RC_heads=[1, 1, 2, 4],
        mlp_ratio=4.,
        NC_group=[1, 32, 64, 128],
        RC_group=[1, 16, 32, 64]),
    decode_head=dict(
        type='DSQNet',
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))