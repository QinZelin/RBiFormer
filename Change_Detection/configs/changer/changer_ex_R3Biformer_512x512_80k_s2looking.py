_base_ = [
    '../_base_/models/R3BiFormer.py',
    '../common/standard_512x512_40k_s2looking.py']
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
#crop_size = (160, 160)
model = dict(
    type='SiamEncoderDecoder',
    #type='EncoderDecoder',
    pretrained='pth/biformer_small_best.pth',
    #pretrained=None,
    backbone=dict(
        type='R3BiFormer',
        depth=[4, 4, 18, 4],
        embed_dim=[64, 128, 256, 512],
        #img_size=[512,512],
        mlp_ratios=[3, 3, 3, 3],
        # use_checkpoint_stages=[0, 1, 2, 3],
        use_checkpoint_stages=[],
        #------------------------------
        n_win=8,
        kv_downsample_mode='identity',
        kv_per_wins=[-1, -1, -1, -1],
        topks=[1, 4, 16, -2],
        side_dwconv=5,
        before_attn_dwconv=3,
        layer_scale_init_value=-1,
        qk_dims=[64, 128, 256, 512],
        head_dim=32,
        param_routing=False, diff_routing=False, soft_routing=False,
        pre_norm=True,
        pe=None,
        out_indices=(0, 1, 2, 3),
        auto_pad=True
        ),
    neck=dict(type='FeatureFusionNeck', policy='concat'),
    decode_head=dict(
        #in_channels=[96, 192, 384, 768],
        in_channels=[64, 128, 256, 512],
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
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
#train_cfg = dict(type='IterBasedTrainLoop', max_iters=2000, val_interval=500)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=8000))
#default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=2000))

# compile = True # use PyTorch 2.x