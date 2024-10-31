_base_ = [
    '../_base_/models/swin-t.py',
    '../common/standard_512x512_80k_levircd.py']

crop_size = (512, 512)

model = dict(
    type='SiamEncoderDecoder',
    pretrained='swin_tiny_patch4_window7_224.pth',
    backbone=dict(
        type='swin',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        type='BMQNet',
        in_channels=[96, 192, 384, 768],
        num_classes=2,
        img_size=512,
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

# optimizer
optimizer=dict(
    type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)

# compile = True # use PyTorch 2.x