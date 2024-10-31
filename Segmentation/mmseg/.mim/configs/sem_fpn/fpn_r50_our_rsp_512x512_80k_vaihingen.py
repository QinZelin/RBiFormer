_base_ = [
    '../_base_/models/fpn_our_r50.py', '../_base_/datasets/vaihingen.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='/public/data3/users/wangdi153/RS_CV/RS_CLS_finetune/output/resnet_50_224/epoch40/ckpt.pth',
    backbone=dict(),
    decode_head=dict(num_classes=5,ignore_index=5), 
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    )
