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
    )