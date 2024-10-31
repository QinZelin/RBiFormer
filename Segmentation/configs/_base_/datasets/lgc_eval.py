# dataset settings
dataset_type = 'LGCDataset'
data_root = '/data/users/qinzilin/dataset/lgc_2/yin_yang_slope/test/'
# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#img_norm_cfg = dict(
#     mean=[15000, 14000, 13000,12000, 11000, 10000], std=[8000, 7000, 6000,5000, 4000, 3000], to_rgb=False)

#东南亚 南美1
# img_norm_cfg = dict(
#      mean=[12033.502, 11220.939, 10339.917,19795.886, 14766.789, 10351.266], std=[4079.835, 4210.605, 4721.364,4498.259, 4605.947, 4001.557], to_rgb=False)
# 南美2
# img_norm_cfg = dict(
#      mean=[8190.8, 7780.9, 7479.9,14147.1, 12224.7, 8711.5], std=[1176.2, 1209.3, 1465.8,2751.9, 2834.8, 2174.2], to_rgb=False)
#中亚
# img_norm_cfg = dict(
#      mean=[10231.5, 10350.0, 10653.9,16278.7, 15682.1, 12805.8], std=[1874.6, 2150.76, 2710.4,3070.3, 3012.0, 2832.3], to_rgb=False)
# # 阴阳坡
img_norm_cfg = dict(
     mean=[8793.3, 8399.6, 8188.6,13150.7, 12351.3, 9670.9], std=[1115.4, 1226.6, 1646.4,2779.0, 2975.7, 2609.6], to_rgb=False)
# 非洲
# img_norm_cfg = dict(
#      mean=[9910.4, 9903.7, 10790.2,14799.8, 15658.8, 13058.1], std=[1185.9, 1389.7, 2009.9,2513.8, 3541.2, 3158.6], to_rgb=False)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile_multispectral'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=(512, 512), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='RandomFlip', prob=0),
    #dict(type='PhotoMetricDistortion'),
    dict(type='Normalize_multi', **img_norm_cfg),
    dict(type='Pad_multi', size=crop_size, pad_val=0, seg_pad_val=255),
    #dict(type='Pad_multi', size=crop_size, pad_val=0, seg_pad_val=4),
    dict(type='DefaultFormatBundle_multi'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg','img_original']),
]

val_pipeline = [
    dict(type='LoadImageFromFile_multispectral'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize_multi', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img','img_original']),
            dict(type='Collect', keys=['img','img_original']),
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile_multispectral'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=None,
        img_ratios=[1.0],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize_multi', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img','img_original']),
            dict(type='Collect', keys=['img','img_original']),
        ])
]
data = dict(
    #samples_per_gpu=8,
    #workers_per_gpu=8,
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/train',
        ann_dir='ann_dir/train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='img_dir/val',
        ann_dir='ann_dir/val',
        pipeline=test_pipeline))
