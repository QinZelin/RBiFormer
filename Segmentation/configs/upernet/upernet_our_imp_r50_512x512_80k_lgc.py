_base_ = [
    '../_base_/models/upernet_our_r50.py', '../_base_/datasets/lgc.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
model = dict(
    pretrained='/data/users/qinzilin/code1/rsfm_seg/pth/resnet50-19c8e357.pth',
    #pretrained=None,
    backbone=dict(),
    #decode_head=dict(num_classes=3,ignore_index=3),
    #auxiliary_head=dict(num_classes=3,ignore_index=3)
    decode_head = dict(num_classes=5),
    auxiliary_head = dict(num_classes=5)
    )
runner = dict(type='IterBasedRunner', max_iters=20000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric='mIoU', pre_eval=True)