"""
Core of BiFormer, Bi-Level Routing Attention.

To be refactored.

author: ZHU Lei
github: https://github.com/rayleizhu
email: ray.leizhu@outlook.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class TopkRouting(nn.Module):
    """
    differentiable topk routing with scaling
    Args:
        qk_dim: int, feature dimension of query and key
        topk: int, the 'topk'
        qk_scale: int or None, temperature (multiply) of softmax activation
        with_param: bool, wether inorporate learnable params in routing unit
        diff_routing: bool, wether make routing differentiable
        soft_routing: bool, wether make output value multiplied by routing weights
    """
    def __init__(self, qk_dim, topk=4, qk_scale=None, param_routing=False, diff_routing=False):
        super().__init__()
        self.topk = topk
        self.qk_dim = qk_dim
        self.scale = qk_scale or qk_dim ** -0.5
        self.diff_routing = diff_routing
        # TODO: norm layer before/after linear?
        self.emb = nn.Linear(qk_dim, qk_dim) if param_routing else nn.Identity()
        # routing activation
        self.routing_act = nn.Softmax(dim=-1)
    
    def forward(self, query:Tensor, key:Tensor)->Tuple[Tensor]:
        """
        Args:
            q, k: (n, p^2, c) tensor
        Return:
            r_weight, topk_index: (n, p^2, topk) tensor
        """
        if not self.diff_routing:
            query, key = query.detach(), key.detach()
        query_hat, key_hat = self.emb(query), self.emb(key) # per-window pooling -> (n, p^2, c) 
        attn_logit = (query_hat*self.scale) @ key_hat.transpose(-2, -1) # (n, p^2, p^2)
        topk_attn_logit, topk_index = torch.topk(attn_logit, k=self.topk, dim=-1) # (n, p^2, k), (n, p^2, k)
        r_weight = self.routing_act(topk_attn_logit) # (n, p^2, k)
        
        return r_weight, topk_index
        

class KVGather(nn.Module):
    def __init__(self, mul_weight='none'):
        super().__init__()
        assert mul_weight in ['none', 'soft', 'hard']
        self.mul_weight = mul_weight

    def forward(self, r_idx:Tensor, r_weight:Tensor, kv:Tensor):
        """
        r_idx: (n, p^2, topk) tensor
        r_weight: (n, p^2, topk) tensor
        kv: (n, p^2, w^2, c_kq+c_v)

        Return:
            (n, p^2, topk, w^2, c_kq+c_v) tensor
        """
        # select kv according to routing index
        n, p2, w2, c_kv = kv.size()
        topk = r_idx.size(-1)
        # print(r_idx.size(), r_weight.size())
        # FIXME: gather consumes much memory (topk times redundancy), write cuda kernel? 
        topk_kv = torch.gather(kv.view(n, 1, p2, w2, c_kv).expand(-1, p2, -1, -1, -1), # (n, p^2, p^2, w^2, c_kv) without mem cpy
                                dim=2,
                                index=r_idx.view(n, p2, topk, 1, 1).expand(-1, -1, -1, w2, c_kv) # (n, p^2, k, w^2, c_kv)
                               )

        if self.mul_weight == 'soft':
            topk_kv = r_weight.view(n, p2, topk, 1, 1) * topk_kv # (n, p^2, k, w^2, c_kv)
        elif self.mul_weight == 'hard':
            raise NotImplementedError('differentiable hard routing TBA')
        # else: #'none'
        #     topk_kv = topk_kv # do nothing

        return topk_kv

class QKVLinear(nn.Module):
    def __init__(self, dim, qk_dim, bias=True):
        super().__init__()
        self.dim = dim
        self.qk_dim = qk_dim
        self.qkv = nn.Linear(dim, qk_dim + qk_dim + dim, bias=bias)
    
    def forward(self, x):
        # q, kv = self.qkv(x).split([self.qk_dim, self.qk_dim+self.dim], dim=-1)
        # return q, kv
        q, k, v = self.qkv(x).split([self.qk_dim, self.qk_dim, self.dim], dim=-1)
        return q, k, v

class BiLevelRoutingAttention(nn.Module):
    """
    n_win: number of windows in one side (so the actual number of windows is n_win*n_win)
    kv_per_win: for kv_downsample_mode='ada_xxxpool' only, number of key/values per window. Similar to n_win, the actual number is kv_per_win*kv_per_win.
    topk: topk for window filtering
    param_attention: 'qkvo'-linear for q,k,v and o, 'none': param free attention
    param_routing: extra linear for routing
    diff_routing: wether to set routing differentiable
    soft_routing: wether to multiply soft routing weights 
    """
    def __init__(self, dim, num_heads=8, n_win=7, qk_dim=None, qk_scale=None,img_size=[224,224],
                 kv_per_win=4, kv_downsample_ratio=4, kv_downsample_kernel=None, kv_downsample_mode='identity',
                 topk=4, param_attention="qkvo", param_routing=False, diff_routing=False, soft_routing=False, side_dwconv=3,
                 auto_pad=False):
        super().__init__()
        # local attention setting
        self.dim = dim
        self.n_win = n_win  # Wh, Ww
        self.num_heads = num_heads
        self.qk_dim = qk_dim or dim
        assert self.qk_dim % num_heads == 0 and self.dim % num_heads==0, 'qk_dim and dim must be divisible by num_heads!'
        self.scale = qk_scale or self.qk_dim ** -0.5


        ################side_dwconv (i.e. LCE in ShuntedTransformer)###########
        self.lepe = nn.Conv2d(dim, dim, kernel_size=side_dwconv, stride=1, padding=side_dwconv//2, groups=dim) if side_dwconv > 0 else \
                    lambda x: torch.zeros_like(x)
        # self.lepe = nn.Sequential(
        #                     nn.Conv2d(dim, 3*dim, side_dwconv, 1, side_dwconv//2, 1, groups=dim),
        #                     nn.BatchNorm2d(3*dim),
        #                     nn.SiLU(inplace=True),
        #                     nn.Conv2d(3*dim, dim, side_dwconv, 1, side_dwconv//2, 1, groups=dim),
        #                     nn.BatchNorm2d(dim),
        #                     nn.SiLU(inplace=True),
        #                     nn.Conv2d(dim, dim, side_dwconv, 1, side_dwconv//2, 1, groups=dim),
        #                     )
        
        ################ global routing setting #################
        self.topk = topk
        self.param_routing = param_routing
        self.diff_routing = diff_routing
        self.soft_routing = soft_routing
        # router
        assert not (self.param_routing and not self.diff_routing) # cannot be with_param=True and diff_routing=False
        self.router = TopkRouting(qk_dim=self.qk_dim,
                                  qk_scale=self.scale,
                                  topk=self.topk,
                                  diff_routing=self.diff_routing,
                                  param_routing=self.param_routing)
        if self.soft_routing: # soft routing, always diffrentiable (if no detach)
            mul_weight = 'soft'
        elif self.diff_routing: # hard differentiable routing
            mul_weight = 'hard'
        else:  # hard non-differentiable routing
            mul_weight = 'none'
        self.kv_gather = KVGather(mul_weight=mul_weight)

        # qkv mapping (shared by both global routing and local attention)
        self.param_attention = param_attention
        if self.param_attention == 'qkvo':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Linear(dim, dim)
        elif self.param_attention == 'qkv':
            self.qkv = QKVLinear(self.dim, self.qk_dim)
            self.wo = nn.Identity()
        else:
            raise ValueError(f'param_attention mode {self.param_attention} is not surpported!')
        
        self.kv_downsample_mode = kv_downsample_mode
        self.kv_per_win = kv_per_win
        self.kv_downsample_ratio = kv_downsample_ratio
        self.kv_downsample_kenel = kv_downsample_kernel
        if self.kv_downsample_mode == 'ada_avgpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveAvgPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'ada_maxpool':
            assert self.kv_per_win is not None
            self.kv_down = nn.AdaptiveMaxPool2d(self.kv_per_win)
        elif self.kv_downsample_mode == 'maxpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.MaxPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'avgpool':
            assert self.kv_downsample_ratio is not None
            self.kv_down = nn.AvgPool2d(self.kv_downsample_ratio) if self.kv_downsample_ratio > 1 else nn.Identity()
        elif self.kv_downsample_mode == 'identity': # no kv downsampling
            self.kv_down = nn.Identity()
        elif self.kv_downsample_mode == 'fracpool':
            # assert self.kv_downsample_ratio is not None
            # assert self.kv_downsample_kenel is not None
            # TODO: fracpool
            # 1. kernel size should be input size dependent
            # 2. there is a random factor, need to avoid independent sampling for k and v 
            raise NotImplementedError('fracpool policy is not implemented yet!')
        elif kv_downsample_mode == 'conv':
            # TODO: need to consider the case where k != v so that need two downsample modules
            raise NotImplementedError('conv policy is not implemented yet!')
        else:
            raise ValueError(f'kv_down_sample_mode {self.kv_downsaple_mode} is not surpported!')

        # softmax for local attention
        self.attn_act = nn.Softmax(dim=-1)

        self.auto_pad=auto_pad




        # -----------
        # 制作各个像素点的坐标
        # 当h、w不相等时，下面代码可能会出问题 211行 h w 反
        self.img_size=img_size
        h, w = self.img_size


        self.pad_r = (self.n_win - w % self.n_win) % self.n_win
        self.pad_b = (self.n_win - h % self.n_win) % self.n_win
        # self.pad_b = (self.n_win - w % self.n_win) % self.n_win
        # self.pad_r = (self.n_win - h % self.n_win) % self.n_win

        h, w = h + self.pad_r, w +self.pad_b
        self.img_h=h
        self.img_w=w
        assert h==w
        # 窗口大小
        self.ws=h//self.n_win

        # 得到偏移参数
        self.sampling_offsets = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.ws, stride=self.ws),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2, kernel_size=1, stride=1)
        )
        self.sampling_scales = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.ws, stride=self.ws),
            nn.LeakyReLU(),
            nn.Conv2d(dim, 2, kernel_size=1, stride=1)
        )

        image_reference_w = torch.linspace(-1, 1, w)
        image_reference_h = torch.linspace(-1, 1, h)
        image_reference = torch.stack(torch.meshgrid(image_reference_w, image_reference_h), 0).permute(0, 2,1).unsqueeze(0)  # 2, h, w
        window_reference = nn.functional.avg_pool2d(image_reference, kernel_size=self.ws)
        window_num_h, window_num_w = window_reference.shape[-2:]
        assert window_num_h==n_win
        assert window_num_w == n_win
        window_reference = window_reference.reshape(1, 2, window_num_h, 1, window_num_w, 1)

        base_coords_h = torch.arange(self.ws) * 2 * self.ws / self.ws / (h - 1)
        base_coords_h = (base_coords_h - base_coords_h.mean())
        base_coords_w = torch.arange(self.ws) * 2 * self.ws / self.ws / (w - 1)
        base_coords_w = (base_coords_w - base_coords_w.mean())

        expanded_base_coords_h = base_coords_h.unsqueeze(dim=0).repeat(window_num_h, 1)
        assert expanded_base_coords_h.shape[0] == window_num_h
        assert expanded_base_coords_h.shape[1] == self.ws
        expanded_base_coords_w = base_coords_w.unsqueeze(dim=0).repeat(window_num_w, 1)
        assert expanded_base_coords_w.shape[0] == window_num_w
        assert expanded_base_coords_w.shape[1] == self.ws
        expanded_base_coords_h = expanded_base_coords_h.reshape(-1)
        expanded_base_coords_w = expanded_base_coords_w.reshape(-1)
        coords = torch.stack(torch.meshgrid(expanded_base_coords_w, expanded_base_coords_h), 0).permute(0, 2,
                                                                                                        1).reshape(1, 2,
                                                                                                                   window_num_h,
                                                                                                                   self.ws,
                                                                                                                   window_num_w,
                                                                                                                   self.ws)
        self.base_coords = (window_reference + coords).cuda()
        # coords [1,2,n_win,ws,n_win,ws]
        self.coords = coords.cuda()


    def forward(self, x, ret_attn_mask=False):
        """
        x: NHWC tensor

        Return:
            NHWC tensor
        """
         # NOTE: use padding for semantic segmentation
        ###################################################
        if self.auto_pad:
            N, H_in, W_in, C = x.size()

            pad_l = pad_t = 0
            # pad_r = (self.n_win - W_in % self.n_win) % self.n_win
            # pad_b = (self.n_win - H_in % self.n_win) % self.n_win
            pad_r=self.pad_r
            pad_b=self.pad_b
            x = F.pad(x, (0, 0, # dim=-1
                          pad_l, pad_r, # dim=-2
                          pad_t, pad_b)) # dim=-3
            _, H, W, _ = x.size() # padded size
        else:
            N, H, W, C = x.size()
            assert H%self.n_win == 0 and W%self.n_win == 0 #


        ###################################################
        # 窗口大小不同 像素点的偏移坐标，实现跨区域注意力
        assert H==self.img_h
        assert W == self.img_w

        x=x.permute(0,3,1,2)
        window_num_h, window_num_w = self.base_coords.shape[-4], self.base_coords.shape[-2]
        coords = self.base_coords.repeat(N, 1, 1, 1, 1, 1)
        sampling_offsets = self.sampling_offsets(x)
        #num_predict_total = N
        sampling_offsets = sampling_offsets.reshape(N, 2, window_num_h, window_num_w)
        # sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (W // self.ws)
        # sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (H // self.ws)
        sampling_offsets[:, 0, ...] = sampling_offsets[:, 0, ...] / (self.n_win)
        sampling_offsets[:, 1, ...] = sampling_offsets[:, 1, ...] / (self.n_win)
        assert self.n_win==window_num_w
        sampling_scales = self.sampling_scales(x)  # B, heads*2, h // window_size, w // window_size
        sampling_scales = sampling_scales.reshape(N, 2, window_num_h, window_num_w)

        coords = coords + self.coords * sampling_scales[:, :, :, None, :, None] + sampling_offsets[:, :, :, None, :,None]
        # sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(N, self.ws * window_num_h,
        #                                                          self.ws * window_num_w, 2)
        sample_coords = coords.permute(0, 2, 3, 4, 5, 1).reshape(N, H,W, 2)
        x = x.permute(0, 2, 3, 1)

        # patchify, (n, p^2, w, w, c), keep 2d window as we need 2d pooling to reduce kv size
        x = rearrange(x, "n (j h) (i w) c -> n (j i) h w c", j=self.n_win, i=self.n_win)

        #################qkv projection###################
        # q: (n, p^2, w, w, c_qk)
        # kv: (n, p^2, w, w, c_qk+c_v)
        # NOTE: separte kv if there were memory leak issue caused by gather

        #q, kv = self.qkv(x)
        q, k, v = self.qkv(x)

        # -----
        # 对 k v 矩阵进行不同大小的窗口采集
        #kv = rearrange(kv, "n (j i) h w c -> n c (j h) (i w)", j=self.n_win, i=self.n_win) # N 2C H W
        k = rearrange(k, "n (j i) h w c -> n c (j h) (i w)", j=self.n_win, i=self.n_win)  # N C H W
        v = rearrange(v, "n (j i) h w c -> n c (j h) (i w)", j=self.n_win, i=self.n_win)  # N C H W
        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        #lepe = self.lepe(kv[:,self.qk_dim:,:,:].contiguous())
        lepe = self.lepe(v.contiguous())
        lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        # kv_grid=kv.clone()
        # kv_grid[:,0:self.dim,:,:]=F.grid_sample(
        #                 kv[:,0:self.dim,:,:],
        #                 grid=sample_coords, padding_mode='zeros', align_corners=True
        #                 )
        # kv_grid[:,self.dim:,:,:]=F.grid_sample(
        #                 kv[:,self.dim:,:,:],
        #                 grid=sample_coords, padding_mode='zeros', align_corners=True
        #                 )
        # kv=kv_grid
        k = F.grid_sample(
            k,
            grid=sample_coords, padding_mode='zeros', align_corners=True
        )
        v = F.grid_sample(
            v,
            grid=sample_coords, padding_mode='zeros', align_corners=True
        )
        #kv = rearrange(kv, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)  # (n, p^2, w, w, c_qk+c_v)
        k = rearrange(k, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)
        v = rearrange(v, "n c (j h) (i w) -> n (j i) h w c", j=self.n_win, i=self.n_win)

        # pixel-wise qkv
        # q_pix: (n, p^2, w^2, c_qk)
        # kv_pix: (n, p^2, h_kv*w_kv, c_qk+c_v)
        q_pix = rearrange(q, 'n p2 h w c -> n p2 (h w) c')
        # kv_pix = self.kv_down(rearrange(kv, 'n p2 h w c -> (n p2) c h w'))
        # kv_pix = rearrange(kv_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
        k_pix = self.kv_down(rearrange(k, 'n p2 h w c -> (n p2) c h w'))
        k_pix = rearrange(k_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)
        v_pix = self.kv_down(rearrange(v, 'n p2 h w c -> (n p2) c h w'))
        v_pix = rearrange(v_pix, '(n j i) c h w -> n (j i) (h w) c', j=self.n_win, i=self.n_win)


        #q_win, k_win = q.mean([2, 3]), kv[..., 0:self.qk_dim].mean([2, 3]) # window-wise qk, (n, p^2, c_qk), (n, p^2, c_qk)
        q_win, k_win = q.mean([2, 3]), k.mean([2, 3])
        ##################side_dwconv(lepe)##################
        # NOTE: call contiguous to avoid gradient warning when using ddp
        # lepe = self.lepe(rearrange(kv[..., self.qk_dim:], 'n (j i) h w c -> n c (j h) (i w)', j=self.n_win, i=self.n_win).contiguous())
        # lepe = rearrange(lepe, 'n c (j h) (i w) -> n (j h) (i w) c', j=self.n_win, i=self.n_win)

        ############ gather q dependent k/v #################

        r_weight, r_idx = self.router(q_win, k_win) # both are (n, p^2, topk) tensors

        # kv_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=kv_pix) #(n, p^2, topk, h_kv*w_kv, c_qk+c_v)
        # k_pix_sel, v_pix_sel = kv_pix_sel.split([self.qk_dim, self.dim], dim=-1)
        k_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=k_pix)
        v_pix_sel = self.kv_gather(r_idx=r_idx, r_weight=r_weight, kv=v_pix)
        # kv_pix_sel: (n, p^2, topk, h_kv*w_kv, c_qk)
        # v_pix_sel: (n, p^2, topk, h_kv*w_kv, c_v)
        
        ######### do attention as normal ####################
        k_pix_sel = rearrange(k_pix_sel, 'n p2 k w2 (m c) -> (n p2) m c (k w2)', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_kq//m) transpose here?
        v_pix_sel = rearrange(v_pix_sel, 'n p2 k w2 (m c) -> (n p2) m (k w2) c', m=self.num_heads) # flatten to BMLC, (n*p^2, m, topk*h_kv*w_kv, c_v//m)
        q_pix = rearrange(q_pix, 'n p2 w2 (m c) -> (n p2) m w2 c', m=self.num_heads) # to BMLC tensor (n*p^2, m, w^2, c_qk//m)

        # param-free multihead attention
        attn_weight = (q_pix * self.scale) @ k_pix_sel # (n*p^2, m, w^2, c) @ (n*p^2, m, c, topk*h_kv*w_kv) -> (n*p^2, m, w^2, topk*h_kv*w_kv)
        attn_weight = self.attn_act(attn_weight)
        out = attn_weight @ v_pix_sel # (n*p^2, m, w^2, topk*h_kv*w_kv) @ (n*p^2, m, topk*h_kv*w_kv, c) -> (n*p^2, m, w^2, c)
        out = rearrange(out, '(n j i) m (h w) c -> n (j h) (i w) (m c)', j=self.n_win, i=self.n_win,
                        h=H//self.n_win, w=W//self.n_win)

        out = out + lepe
        # output linear
        out = self.wo(out)

        # NOTE: use padding for semantic segmentation
        # crop padded region
        if self.auto_pad and (pad_r > 0 or pad_b > 0):
            out = out[:, :H_in, :W_in, :].contiguous()

        if ret_attn_mask:
            return out, r_weight, r_idx, attn_weight
        else:
            return out
