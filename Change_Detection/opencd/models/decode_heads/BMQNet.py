# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from mmcv.cnn import Conv2d, ConvModule, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmengine.model import BaseModule, Sequential
from torch.nn import functional as F

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import resize
from opencd.registry import MODELS
from ..necks.feature_fusion import FeatureFusionNeck
class LMLP(BaseModule):
    def __init__(self,
                 num_groups=32,
                 num_channels=512):
        super(LMLP, self).__init__()
        self.gn1 = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.conv1=nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1,padding=0)
        self.gn2 = nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
        self.LLMLP = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.ReLU(),
            nn.Linear(num_channels, num_channels),
        )
    def forward(self,x):
        x=self.conv1(self.gn1(x))+x
        x=self.gn2(x)
        x= rearrange(x, 'b c h w-> b h w c')
        x=self.LLMLP(x)+x
        x = rearrange(x, 'b h w c-> b c h w')
        return x
class CFP(BaseModule):
    def __init__(self,
                 in_channels,
                 norm_cfg,
                 img_size,
                 act_cfg):
        super(CFP, self).__init__()
        self.in_channels=in_channels
        self.mlp_channels=128
        self.mlpCov=nn.Conv2d(self.in_channels[3], self.mlp_channels, kernel_size=1, stride=1,padding=0)
        self.MLP=LMLP(num_groups=self.mlp_channels//16,num_channels=self.mlp_channels)
        self.norm_cfg=norm_cfg
        self.act_cfg=act_cfg

        h=img_size//32
        w = img_size // 32
        assert h==w
        kernel_size = stride = h // 2
        # the numbers of parameters of model is bigger
        self.query_bias_pool=ConvModule(
                        in_channels=self.mlp_channels,
                        out_channels=self.mlp_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        norm_cfg=self.norm_cfg)
        #self.query_bias_pool=nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.query_bias_conv = ConvModule(
                     in_channels=self.mlp_channels,
                     out_channels=self.mlp_channels,
                     kernel_size=1,
                     stride=1,
                     norm_cfg=self.norm_cfg,
                     act_cfg=self.act_cfg)

        self.convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            if i==3:
               self.convs.append(
                    ConvModule(
                     in_channels=self.in_channels[i]+self.mlp_channels,
                     out_channels=self.mlp_channels,
                     kernel_size=1,
                     stride=1,
                     norm_cfg=self.norm_cfg,
                     act_cfg=self.act_cfg))
            elif i==0:
               self.convs.append(
                    ConvModule(
                        in_channels=self.in_channels[i] + self.mlp_channels+self.mlp_channels,
                        out_channels=self.mlp_channels //2,
                        kernel_size=1,
                        stride=1,
                        norm_cfg=self.norm_cfg))
            else:
                self.convs.append(
                    ConvModule(
                        in_channels=self.in_channels[i] + self.mlp_channels + self.mlp_channels,
                        out_channels=self.mlp_channels,
                        kernel_size=1,
                        stride=1,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))



    def forward(self,inputs):

        x_mlp=self.mlpCov(inputs[3])
        x_mlp=self.MLP(x_mlp)

        query_bias=self.query_bias_conv(x_mlp)
        query_bias=self.query_bias_pool(query_bias)
        # 4.
        out=torch.cat((x_mlp,inputs[3]), dim=1)
        out=self.convs[3](out)

        # 3.
        h,w=inputs[2].shape[2:]
        out = F.interpolate(out, size=(h, w), mode='bilinear',
                                     align_corners=False)
        x_mlp_interpolate=F.interpolate(x_mlp, size=(h, w), mode='bilinear',
                                     align_corners=False)
        out=torch.cat((x_mlp_interpolate,out,inputs[2]),dim=1)
        out=self.convs[2](out)

        #2.
        h,w=inputs[1].shape[2:]
        out = F.interpolate(out, size=(h, w), mode='bilinear',
                                     align_corners=False)
        x_mlp_interpolate=F.interpolate(x_mlp, size=(h, w), mode='bilinear',
                                     align_corners=False)
        out=torch.cat((x_mlp_interpolate,out,inputs[1]),dim=1)
        out=self.convs[1](out)

        #1.
        h,w=inputs[0].shape[2:]
        out = F.interpolate(out, size=(h, w), mode='bilinear',
                                     align_corners=False)
        x_mlp_interpolate=F.interpolate(x_mlp, size=(h, w), mode='bilinear',
                                     align_corners=False)
        out=torch.cat((x_mlp_interpolate,out,inputs[0]),dim=1)
        out=self.convs[0](out)

        return out, query_bias
class FDAF(BaseModule):
    """Flow Dual-Alignment Fusion Module.

    Args:
        in_channels (int): Input channels of features.
        conv_cfg (dict | None): Config of conv layers.
            Default: None
        norm_cfg (dict | None): Config of norm layers.
            Default: dict(type='BN')
        act_cfg (dict): Config of activation layers.
            Default: dict(type='ReLU')
    """

    def __init__(self,
                 in_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='IN'),
                 act_cfg=dict(type='GELU')):
        super(FDAF, self).__init__()
        self.in_channels = in_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        # TODO
        conv_cfg=None
        norm_cfg=dict(type='IN')
        act_cfg=dict(type='GELU')
        
        kernel_size = 5
        self.flow_make = Sequential(
            nn.Conv2d(in_channels*2, in_channels*2, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=True, groups=in_channels*2),
            nn.InstanceNorm2d(in_channels*2),
            nn.GELU(),
            nn.Conv2d(in_channels*2, 4, kernel_size=1, padding=0, bias=False),
        )

    def forward(self, x1, x2, fusion_policy=None):
        """Forward function."""

        output = torch.cat([x1, x2], dim=1)
        flow = self.flow_make(output)
        f1, f2 = torch.chunk(flow, 2, dim=1)
        x1_feat = self.warp(x1, f1) - x2
        x2_feat = self.warp(x2, f2) - x1
        
        if fusion_policy == None:
            return x1_feat, x2_feat
        
        output = FeatureFusionNeck.fusion(x1_feat, x2_feat, fusion_policy)
        return output

    @staticmethod
    def warp(x, flow):
        n, c, h, w = x.size()

        norm = torch.tensor([[[[w, h]]]]).type_as(x).to(x.device)
        col = torch.linspace(-1.0, 1.0, h).view(-1, 1).repeat(1, w)
        row = torch.linspace(-1.0, 1.0, w).repeat(h, 1)
        grid = torch.cat((row.unsqueeze(2), col.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(x).to(x.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(x, grid, align_corners=True)
        return output


class MixFFN(BaseModule):
    """An implementation of MixFFN of Segformer. \
        Here MixFFN is uesd as projection head of Changer.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 act_cfg=dict(type='GELU'),
                 ffn_drop=0.,
                 dropout_layer=None,
                 init_cfg=None):
        super(MixFFN, self).__init__(init_cfg)

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        in_channels = embed_dims
        fc1 = Conv2d(
            in_channels=in_channels,
            out_channels=feedforward_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        # 3x3 depth wise conv to provide positional encode information
        pe_conv = Conv2d(
            in_channels=feedforward_channels,
            out_channels=feedforward_channels,
            kernel_size=3,
            stride=1,
            padding=(3 - 1) // 2,
            bias=True,
            groups=feedforward_channels)
        fc2 = Conv2d(
            in_channels=feedforward_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            bias=True)
        drop = nn.Dropout(ffn_drop)
        layers = [fc1, pe_conv, self.activate, drop, fc2, drop]
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


@MODELS.register_module()
class BMQNet(BaseDecodeHead):

    def __init__(self, img_size=512,interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)
        assert num_inputs == len(self.in_index)

        self.neck_layer = FDAF(in_channels=self.channels // 2)
        
        # projection head
        self.discriminator = MixFFN(
            embed_dims=self.channels,
            feedforward_channels=self.channels,
            ffn_drop=0.,
            dropout_layer=dict(type='DropPath', drop_prob=0.),
            act_cfg=dict(type='GELU'))


        self.change_areas =self.channels
        self.reference_area1 = nn.Embedding(self.change_areas, 4)
        self.reference_area2 = nn.Embedding(self.change_areas, 4)
        self.point1 = nn.Sequential(
            nn.Conv2d(self.channels // 2, self.channels // 8, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
        )
        self.point2 = nn.Sequential(
            nn.Linear(self.channels // 8 * 3 * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )
        nn.init.constant_(self.point2[-1].weight.data, 0)
        nn.init.constant_(self.point2[-1].bias.data, 0)


        self.norm_conv = ConvModule(
            in_channels=self.channels //2,
            out_channels=self.channels // 2,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

        self.CFP=CFP(in_channels=self.in_channels,norm_cfg=self.norm_cfg,act_cfg=self.act_cfg,img_size=img_size)


    def box_cxcywh_to_xyxy(slef,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)
    def BMQ(self,x1,x2,reference_area1,reference_area2,query_bias1,query_bias2):
        b, c, h, w = x1.shape
        reference_area1=reference_area1.unsqueeze(0).repeat(b, 1, 1)
        query_bias1=query_bias1.view(b,self.change_areas,4)
        reference_area1=reference_area1+query_bias1
        reference_area1 = reference_area1.sigmoid() # [batch_size, 256, 4]

        reference_area1_xyxy = self.box_cxcywh_to_xyxy(reference_area1)
        reference_area1_xyxy[:, :, 0] *= w
        reference_area1_xyxy[:, :, 1] *= h
        reference_area1_xyxy[:, :, 2] *= w
        reference_area1_xyxy[:, :, 3] *= h

        out1_content = torchvision.ops.roi_align(
            x2,
            list(torch.unbind(reference_area1_xyxy, dim=0)),
            output_size=(3, 3),
            spatial_scale=1.0,
            aligned=True)  # (bs * 256, c, 3, 3)

        out1_content_points = torchvision.ops.roi_align(
            x2,
            list(torch.unbind(reference_area1_xyxy, dim=0)),
            output_size=(3, 3),
            spatial_scale=1.0,
            aligned=True)  # (bs * 256, c, 3, 3)
        out1_content_index = out1_content_points.view(b * self.change_areas, -1, 3, 3)
        points1 = self.point1(out1_content_index)  # (bs * 128, c//4, 3, 3)
        points1 = points1.reshape(b * self.change_areas, -1)  # (bs * 128, c//4 * 3 * 3)
        points1 = self.point2(points1)  # (bs * 128, 2)
        points1 = points1.view(b * self.change_areas, 1,1,2).tanh()

        out1_content = F.grid_sample(out1_content, points1, padding_mode="zeros", align_corners=False).view(b * self.change_areas,-1)
        out1_content = out1_content.view(b, self.change_areas, -1).permute(0, 2, 1)# (b,128,c) ->(b,c,128)
        #print(out1_content.shape)
        out1_content=out1_content.view(b, c, int(self.change_areas//16),16)  # (b,c, self.change_areas//2,self.change_areas//2)
        out1_content = F.interpolate(out1_content, size=(h, w), mode='bilinear',
                                         align_corners=False)
        out1_content = x1+out1_content



        reference_area2 = reference_area2.unsqueeze(0).repeat(b, 1, 1)
        query_bias2 = query_bias2.view(b,self.change_areas, 4)
        reference_area2=reference_area2+query_bias2
        reference_area2 = reference_area2.sigmoid()  # [batch_size, 128, 4]

        reference_area2_xyxy = self.box_cxcywh_to_xyxy(reference_area2)
        reference_area2_xyxy[:, :, 0] *= w
        reference_area2_xyxy[:, :, 1] *= h
        reference_area2_xyxy[:, :, 2] *= w
        reference_area2_xyxy[:, :, 3] *= h

        out2_content = torchvision.ops.roi_align(
            x1,
            list(torch.unbind(reference_area2_xyxy, dim=0)),
            output_size=(3, 3),
            spatial_scale=1.0,
            aligned=True)  # (bs * 128, c, 3, 3)

        out2_content_points = torchvision.ops.roi_align(
            x1,
            list(torch.unbind(reference_area2_xyxy, dim=0)),
            output_size=(3, 3),
            spatial_scale=1.0,
            aligned=True)  # (bs * 128, c, 3, 3)
        out2_content_index = out2_content_points.view(b * self.change_areas, -1, 3, 3)
        points2 = self.point1(out2_content_index)  # (bs * 128, c//4, 3, 3)
        points2 = points2.reshape(b * self.change_areas, -1)  # (bs * 128, c//4 * 3 * 3)
        points2 = self.point2(points2)  # (bs * 128, 2)
        points2 = points2.view(b * self.change_areas, 1, 1, 2).tanh()

        out2_content = F.grid_sample(out2_content, points2, padding_mode="zeros", align_corners=False).view(
            b * self.change_areas, -1)
        out2_content = out2_content.view(b, self.change_areas, -1).permute(0, 2, 1)  # (b,128,c) ->(b,c,128)
        # print(out1_content.shape)
        out2_content = out2_content.view(b, c, int(self.change_areas//16),
                                         16)  # (b,c, self.change_areas**0.5,self.change_areas**0.5)
        out2_content = F.interpolate(out2_content, size=(h, w), mode='bilinear',
                                     align_corners=False)
        out2_content = x2 + out2_content


        return out1_content,out2_content
    



    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        inputs1 = []
        inputs2 = []
        for input in inputs:
            f1, f2 = torch.chunk(input, 2, dim=1)
            inputs1.append(f1)
            inputs2.append(f2)

        # DSR-FPN
        out1,query_bias1=self.CFP(inputs1)
        out2,query_bias2 = self.CFP(inputs2)
        b, c, h, w = out1.shape

        out1,out2=self.BMQ(out1,out2,self.reference_area1.weight,self.reference_area2.weight,query_bias1,query_bias2)

        out1=self.norm_conv(out1)
        out2 = self.norm_conv(out2)

        out = self.neck_layer(out1, out2, 'concat')
        out = self.discriminator(out)
        out = self.cls_seg(out)

        return out
