# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
#from .swin_mlp import SwinMLP
from .ViTAE_Window_NoShift.models import ViTAE_Window_NoShift_12_basic_stages4_14
from .resnet import resnet50
from R3BiFormer import R3BiFormer
def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'resnet':

        model = resnet50(num_classes=config.MODEL.NUM_CLASSES)

        print('Using ResNet50!')

    elif model_type == 'vitae_win':

        model = ViTAE_Window_NoShift_12_basic_stages4_14(img_size=config.DATA.IMG_SIZE, num_classes=config.MODEL.NUM_CLASSES, window_size=7)

        print('Using VitAE_v2!')

    elif model_type == 'r3biformer':

        model = R3BiFormer(depth=[4, 4, 18, 4],
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
        auto_pad=True)

        print('Using R3BiFormer!')
        
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
