_base_ = [
    '../_base_/models/vitaev2.py',
    '../common/standard_512x512_80k_s2looking.py']

crop_size = (512, 512)
model = dict(
    type='SiamEncoderDecoder',
    # type='DIEncoderDecoder',
    # type='EncoderDecoder',
    # pretrained='rsp-swin-t-ckpt.pth',
    pretrained='ViTAEv2-S.pth.tar',
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
        RC_group=[1, 16, 32, 64],
        img_size=512,
        window_size=7,
        drop_path_rate=0.3
    ),
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        type='DSQNet',
        in_channels=[64, 128, 256, 512],
        num_classes=2,
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
    )

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(pipeline=train_pipeline))

optimizer=dict(
    type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=120000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# learning policy
# training schedule for 80k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=120000, val_interval=4000)
#train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=500)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=4000))
# learning policy
# training schedule for 80k
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=8000)
# #train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=500)
# default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=8000))
#default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2000))

# compile = True # use PyTorch 2.x