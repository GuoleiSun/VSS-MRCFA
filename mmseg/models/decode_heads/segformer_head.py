# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead, BaseDecodeHead_clips
from mmseg.models.utils import *
import attr

from IPython import embed
from .pvt.swin_transformer import BasicLayer, BasicLayer_intra_inter
from .pvt.swin_transformer_2d import BasicLayer as BasicLayer_swin_2d
from .pvt.swin_transformer_2d import BasicLayer_cluster, BasicLayer_cluster_mask
from .pvt.focal_transformer import BasicLayer3d as BasicLayer_focal3d
from .pvt.focal_transformer import BasicLayer3d2 as BasicLayer_focal3d2
from .pvt.focal_transformer import BasicLayer3d3 as BasicLayer_focal3d3
from .pvt.focal_transformer import BasicLayer3d3_nochange as BasicLayer_focal3d3_nochange
from .pvt.focal_transformer import BasicLayer3d3_onlycffm as BasicLayer_focal3d3_onlycffm
from .pvt.focal_transformer import BasicLayer3d3_selfatten as BasicLayer_focal3d3_selfatten

import cvbase as cvb
import cv2
from .hypercorre import hypercorre, hypercorre2, hypercorre_topk2, hypercorre_topk1, multi_scale_atten
from fast_pytorch_kmeans import KMeans
from .utils.utils import save_cluster_labels
import time
from ..builder import build_loss
from torchpq.clustering import MultiKMeans
from torchpq.clustering import KMeans as KMeans_torchpq
from torch.nn import functional as F


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            # norm_cfg=dict(type='SyncBN', requires_grad=True)
            norm_cfg=dict(type='GN', num_groups=1)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        print(c1.shape, c2.shape, c3.shape, c4.shape)

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        # print(torch.cuda.memory_allocated(0))

        return x


@HEADS.register_module()
class SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4(BaseDecodeHead_clips):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    use hypercorrection in hsnet
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_clips2_resize_1_8_hypercorrelation2_topk_ensemble4, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        # self.linear_pred2 = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

        self.deco1=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco2=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco3=small_decoder2(embedding_dim,256, self.num_classes)
        self.deco4=small_decoder2(embedding_dim,256, self.num_classes)

        # depths=4
        # self.decoder_swin=BasicLayer_swin_2d(
        #         dim=embedding_dim,
        #         depth=depths,
        #         num_heads=8,
        #         window_size=(2,7,7),
        #         mlp_ratio=4.,
        #         qkv_bias=True,
        #         qk_scale=None,
        #         drop=0.,
        #         attn_drop=0.,
        #         drop_path=0.,
        #         norm_layer=nn.LayerNorm,
        #         downsample=None,
        #         use_checkpoint=False)

        self.hypercorre_module=hypercorre_topk2(dim=self.in_channels)

        reference_size="1_32"   ## choices: 1_32, 1_16
        if reference_size=="1_32":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=8, stride=8)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=4, stride=4)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=2, stride=2)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=4, stride=4)
        elif reference_size=="1_16":
            # self.sr1 = nn.Conv2d(c1_in_channels, c1_in_channels, kernel_size=4, stride=4)
            self.sr2 = nn.Conv2d(c2_in_channels, c2_in_channels, kernel_size=2, stride=2)
            self.sr3 = nn.Conv2d(c3_in_channels, c3_in_channels, kernel_size=1, stride=1)
            self.sr1_feat=nn.Conv2d(embedding_dim, embedding_dim, kernel_size=2, stride=2)

        self.self_ensemble2=True

    def forward(self, inputs, batch_size=None, num_clips=None):
        start_time=time.time()
        if self.training:
            assert self.num_clips==num_clips
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _, _, h, w=_c.shape
        x = self.dropout(_c)
        x = self.linear_pred(x)
        x = x.reshape(batch_size, num_clips, -1, h, w)

        # print(x.shape)
        if not self.training and num_clips!=self.num_clips:
        # if not self.training:
            return x[:,-1]

        # if not self.training and num_clips!=self.num_clips:
        #     return x[:,-1]
        # else:
        #     # print(x.shape, num_clips, self.num_clips, self.training)
        #     return x[:,-2]

        start_time1=time.time()
        shape_c1, shape_c2, shape_c3, shape_c4=c1.size()[2:], c2.size()[2:], c3.size()[2:], c4.size()[2:]
        c1=c1.reshape(batch_size, num_clips, -1, c1.shape[-2], c1.shape[-1])
        c2=c2.reshape(batch_size, num_clips, -1, c2.shape[-2], c2.shape[-1])
        c3=c3.reshape(batch_size, num_clips, -1, c3.shape[-2], c3.shape[-1])
        c4=c4.reshape(batch_size, num_clips, -1, c4.shape[-2], c4.shape[-1])
        query_c1, query_c2, query_c3, query_c4=c1[:,:-1], c2[:,:-1], c3[:,:-1], c4[:,:-1]
        
        query_c2=query_c2.reshape(batch_size*(num_clips-1), -1, shape_c2[0], shape_c2[1])
        query_c3=query_c3.reshape(batch_size*(num_clips-1), -1, shape_c3[0], shape_c3[1])

        # query_c1=self.sr1(query_c1)
        query_c2=self.sr2(query_c2)
        query_c3=self.sr3(query_c3)

        # query_c1=query_c1.reshape(batch_size, (num_clips-1), -1, query_c1.shape[-2], query_c1.shape[-1])
        query_c2=query_c2.reshape(batch_size, (num_clips-1), -1, query_c2.shape[-2], query_c2.shape[-1])
        query_c3=query_c3.reshape(batch_size, (num_clips-1), -1, query_c3.shape[-2], query_c3.shape[-1])
        # query_c4=query_c4.reshape(batch_size, (num_clips-1), -1, query_c4.shape[-2], query_c4.shape[-1])

        query_frame=[query_c1, query_c2, query_c3, query_c4]
        supp_frame=[c1[:,-1:], c2[:,-1:], c3[:,-1:], c4[:,-1:]]
        # supp_frame=[c1[-batch_size:].unsqueeze(1), c2[-batch_size:].unsqueeze(1), c3[-batch_size:].unsqueeze(1), c4[-batch_size:].unsqueeze(1)]
        # print([i.shape for i in query_frame])
        # print([i.shape for i in supp_frame])
        start_time11=time.time()
        atten, topk_mask=self.hypercorre_module(query_frame, supp_frame)
        # print(atten.shape, atten.max(), atten.min())
        # exit()
        atten=F.softmax(atten,dim=-1)

        start_time2=time.time()

        h2=int(h/2)
        w2=int(w/2)
        # h3,w3=shape_c3[-2], shape_c3[-1]
        _c2 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)
        _c2_split=_c2.reshape(batch_size, num_clips, -1, h2, w2)
        # _c_further=_c2[:,:-1].reshape(batch_size, num_clips-1, -1, h3*w3)
        _c3=self.sr1_feat(_c2)
        _c3=_c3.reshape(batch_size, num_clips, -1, _c3.shape[-2]*_c3.shape[-1]).transpose(-2,-1)
        # _c_further=_c3[:,:-1].reshape(batch_size, num_clips-1, _c2.shape[-2], _c2.shape[-1], -1)    ## batch_size, num_clips-1, _c2.shape[-2], _c2.shape[-1], c
        _c_further=_c3[:,:-1]        ## batch_size, num_clips-1, _c2.shape[-2]*_c2.shape[-1], c
        # print(_c_further.shape, topk_mask.shape, torch.unique(topk_mask.sum(2)))
        _c_further=_c_further[topk_mask].reshape(batch_size,num_clips-1,-1,_c_further.shape[-1])    ## batch_size, num_clips-1, s, c
        supp_feats=torch.matmul(atten,_c_further)
        supp_feats=supp_feats.transpose(-2,-1).reshape(batch_size, (num_clips-1), -1, h2,w2)
        supp_feats=(torch.chunk(supp_feats, (num_clips-1), dim=1))
        supp_feats=[ii.squeeze(1) for ii in supp_feats]
        supp_feats.append(_c2_split[:,-1])

        outs=supp_feats

        out1=resize(self.deco1(outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out2=resize(self.deco2(outs[1]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out3=resize(self.deco3(outs[2]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4(outs[3]+outs[2]+outs[1]+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        # out4=resize(self.deco4((outs[3]+outs[2]+outs[1])/3.0+outs[0]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)
        out4=resize(self.deco4((outs[0]+outs[1]+outs[2])/3.0+outs[3]), size=(h, w),mode='bilinear',align_corners=False).unsqueeze(1)

        output=torch.cat([x,out1,out2,out3,out4],dim=1)   ## b*(k+k)*124*h*w

        if not self.training:
            # return output.squeeze(1)
            # return torch.cat([x2,x3],1).mean(1)
            return out4.squeeze(1)
            # return out4.squeeze(1)+(out3.squeeze(1)+out2.squeeze(1)+out1.squeeze(1))/3
            # return F.softmax(torch.cat([out1,out2,out3,out4],1),dim=2).sum(1)
            # return torch.cat([out1,out2,out3,out4],1).mean(1)

        return output

        # x2 = self.dropout(supp_feats)
        # x2 = self.linear_pred2(x2)
        # x2=resize(x2, size=(h, w),mode='bilinear',align_corners=False)
        # x2=x2.reshape(batch_size, (num_clips-1), -1, h,w)

        # _c3 = resize(_c, size=(h2,w2),mode='bilinear',align_corners=False)
        # _c3=_c3.reshape(batch_size, num_clips, -1, h2, w2)
        # _c_further2=supp_feats.reshape(batch_size, (num_clips-1), -1, h2,w2).mean(1)+_c3[:,-1]
        # x3 = self.dropout(_c_further2)
        # x3 = self.linear_pred2(x3)
        # x3=resize(x3, size=(h, w),mode='bilinear',align_corners=False)
        # x3=x3.reshape(batch_size, 1, -1, h, w)

        # x=torch.cat([x,x2,x3],1)   ## b*(k+k-1+1)*124*h*w

        # start_time3=time.time()
        # # print(start_time1-start_time, start_time11-start_time1, start_time2-start_time11, start_time3-start_time2)
        # # exit()

        # if not self.training:
        #     return x3.squeeze(1)
        #     # return torch.cat([x2,x3],1).mean(1)

        # return x

   
class small_decoder2(nn.Module):

    def __init__(self,
                 input_dim=256, hidden_dim=256, num_classes=124,dropout_ratio=0.1):
        super().__init__()
        self.hidden_dim=hidden_dim
        self.num_classes=num_classes

        self.smalldecoder=nn.Sequential(
            # ConvModule(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            # ConvModule(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=1, norm_cfg=dict(type='SyncBN', requires_grad=True)),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(hidden_dim, self.num_classes, kernel_size=1)
            )
        # self.dropout=
        
    def forward(self, input):

        output=self.smalldecoder(input)

        return output
