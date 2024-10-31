import time

import torch
from einops import rearrange

from vsa import VSAWindowAttention

# import torch
# import torch.nn as nn
# a=torch.randn((32,64,128,128)).cuda()
# model=VSAWindowAttention(64, out_dim=64, num_heads=8, window_size=7, qkv_bias=True, qk_scale=None,
#                 img_size=(128,128)).cuda()
# b=model(a)
# #b=sampling_offsets(a)
# print(b.shape)

# a=torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
# print(a.shape)
# b=torch.tensor([[1,1],[0,0]])
# print(b.shape)
# c=a*b[:,:,None]
# print(c)
# print(c.shape)
# a=torch.randn((32,3,224,224)).cuda()
# # c=a.permute(0,2,3,1)
# # print(c.shape)
# # b=sampling_offsets(a)
# # print(b.shape)
# current_time=time.time()
# x = a.reshape(96,224,224)
# current_time2=time.time()
# print((current_time2-current_time)*1000)
