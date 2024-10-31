_base_ = [
    '../_base_/models/upernet_our_r50.py', '../_base_/datasets/potsdam.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
model = dict(
    pretrained='/public/data3/users/wangdi153/RS_CV/pretrain_model/seco_resnet50_1m.pth',
    backbone=dict(),
    decode_head=dict(num_classes=5,ignore_index=5), 
    auxiliary_head=dict(num_classes=5,ignore_index=5)
    )
