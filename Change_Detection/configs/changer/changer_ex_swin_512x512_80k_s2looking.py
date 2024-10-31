_base_ = [
    '../_base_/models/changer_r18.py', 
    '../common/standard_512x512_40k_s2looking.py']
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
#crop_size = (160, 160)
model = dict(
    type='SiamEncoderDecoder',
    #type='EncoderDecoder',
    #pretrained='rsp-swin-t-ckpt.pth',
    pretrained=None,
    backbone=dict(
        type='swin',
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        # embed_dim=128,
        # depths=[2, 2, 18, 2],
        # num_heads=[4, 8, 16, 32],
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
        in_channels=[96, 192, 384, 768],
        #in_channels=[128, 256, 512, 1024],
        num_classes=2,
        sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000)),
        # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),

    # auxiliary_head=dict(
    #     type='mmseg.FCNHead',
    #     in_channels=1024,
    #     in_index=2,
    #     channels=256,
    #     num_convs=1,
    #     concat_input=False,
    #     dropout_ratio=0.1,
    #     num_classes=2,
    #     norm_cfg=norm_cfg,
    #     align_corners=False,
    #     loss_decode=dict(
    #         type='mmseg.CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),

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
    dataset=dict(pipeline=train_pipeline))

# learning policy
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1000,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]
# training schedule for 80k
#train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=500)
#default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=8000))
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2000))

# compile = True # use PyTorch 2.x