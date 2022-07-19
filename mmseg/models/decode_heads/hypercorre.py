import torch
import torch.nn as nn
import torch.nn.functional as F
import time

class CenterPivotConv4d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()
        out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y

class CenterPivotConv4d_fast(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d_fast, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
                               bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        bsz, inch, ha, wa, hb, wb = x.size()
        # out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2 = x.transpose(1,3).contiguous().view(-1, inch, hb, wb)  # bsz*wa*ha,inch,hb,wb
        out2 = self.conv2(out2)
        outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        # out2 = out2.view(bsz, wa, ha, outch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
        out2 = out2.view(bsz, wa, ha, outch, o_hb, o_wb).transpose(1,3).contiguous()

        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y

class CenterPivotConv4d_fast2(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d_fast2, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
        #                        bias=bias, padding=padding[:2])
        # self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
        #                        bias=bias, padding=padding[2:])

        # print(type(kernel_size)) # tuple
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size[:2]+(1,), stride=stride[:2]+(1,),
                               bias=bias, padding=padding[:2]+(0,))
        self.conv2 = nn.Conv3d(in_channels, out_channels, (1,)+kernel_size[2:], stride=(1,)+stride[2:],
                               bias=bias, padding=(0,)+padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        if self.stride[2:][-1] > 1:
            print("here")
            out1 = self.prune(x)
        else:
            out1 = x
        bsz, inch, ha, wa, hb, wb = out1.size()
        # out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        out1=out1.reshape(bsz,inch,ha,wa,hb*wb)
        out1 = self.conv1(out1)
        outch, o_ha, o_wa, hb_by_wb = out1.size(-4), out1.size(-3), out1.size(-2), out1.size(-1)
        # out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()
        assert hb_by_wb==hb*wb
        out1=out1.view(bsz,outch,o_ha,o_wa,hb,wb)

        bsz, inch, ha, wa, hb, wb = x.size()
        # out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        out2=x.reshape(bsz, inch, ha*wa, hb, wb)
        out2 = self.conv2(out2)
        outch, ha_by_wa, o_hb, o_wb = out2.size(-4), out2.size(-3), out2.size(-2), out2.size(-1)
        assert ha_by_wa==ha*wa
        # out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        out2=out2.view(bsz,outch,ha,wa,hb,wb)

        assert out1.size()==out2.size()
        if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
            out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
            out2 = out2.squeeze()

        y = out1 + out2
        return y

class CenterPivotConv4d_half(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv4d_half, self).__init__()

        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size[:2], stride=stride[:2],
        #                        bias=bias, padding=padding[:2])
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size[2:], stride=stride[2:],
                               bias=bias, padding=padding[2:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    def prune(self, ct):
        bsz, ch, ha, wa, hb, wb = ct.size()
        if not self.idx_initialized:
            idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
            idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
            self.len_h = len(idxh)
            self.len_w = len(idxw)
            self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
            self.idx_initialized = True
        ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

        return ct_pruned

    def forward(self, x):
        ## x should be size of bsz*s, inch, hb, wb

        # if self.stride[2:][-1] > 1:
        #     out1 = self.prune(x)
        # else:
        #     out1 = x
        # bsz, inch, ha, wa, hb, wb = out1.size()
        # out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        # out1 = self.conv1(out1)
        # outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        # out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        # bsz, inch, ha, wa, hb, wb = x.size()
        bsz_s, inch, hb, wb = x.size()
        out2 = self.conv2(x)

        # out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        # out2 = self.conv2(out2)
        # outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        # out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        # if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
        #     out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
        #     out2 = out2.squeeze()

        # y = out1 + out2
        return out2

class CenterPivotConv3d(nn.Module):
    r""" CenterPivot 4D conv"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(CenterPivotConv3d, self).__init__()

        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size[1:], stride=stride[1:],
                               bias=bias, padding=padding[1:])

        self.stride34 = stride[2:]
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.idx_initialized = False

    # def prune(self, ct):
    #     bsz, ch, ha, wa, hb, wb = ct.size()
    #     if not self.idx_initialized:
    #         idxh = torch.arange(start=0, end=hb, step=self.stride[2:][0], device=ct.device)
    #         idxw = torch.arange(start=0, end=wb, step=self.stride[2:][1], device=ct.device)
    #         self.len_h = len(idxh)
    #         self.len_w = len(idxw)
    #         self.idx = (idxw.repeat(self.len_h, 1) + idxh.repeat(self.len_w, 1).t() * wb).view(-1)
    #         self.idx_initialized = True
    #     ct_pruned = ct.view(bsz, ch, ha, wa, -1).index_select(4, self.idx).view(bsz, ch, ha, wa, self.len_h, self.len_w)

    #     return ct_pruned

    def forward(self, x):
        ## x should be size of bsz*s, inch, hb, wb

        # if self.stride[2:][-1] > 1:
        #     out1 = self.prune(x)
        # else:
        #     out1 = x
        # bsz, inch, ha, wa, hb, wb = out1.size()
        # out1 = out1.permute(0, 4, 5, 1, 2, 3).contiguous().view(-1, inch, ha, wa)
        # out1 = self.conv1(out1)
        # outch, o_ha, o_wa = out1.size(-3), out1.size(-2), out1.size(-1)
        # out1 = out1.view(bsz, hb, wb, outch, o_ha, o_wa).permute(0, 3, 4, 5, 1, 2).contiguous()

        # bsz, inch, ha, wa, hb, wb = x.size()
        bsz, inch, s,  hb, wb = x.size()
        out2 = self.conv2(x)

        # out2 = x.permute(0, 2, 3, 1, 4, 5).contiguous().view(-1, inch, hb, wb)
        # out2 = self.conv2(out2)
        # outch, o_hb, o_wb = out2.size(-3), out2.size(-2), out2.size(-1)
        # out2 = out2.view(bsz, ha, wa, outch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()

        # if out1.size()[-2:] != out2.size()[-2:] and self.padding[-2:] == (0, 0):
        #     out1 = out1.view(bsz, outch, o_ha, o_wa, -1).sum(dim=-1)
        #     out2 = out2.squeeze()

        # y = out1 + out2
        return out2



class HPNLearner(nn.Module):
    def __init__(self, inch):
        super(HPNLearner, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1
        # outch1, outch2, outch3, outch_final = 2,4,4,1
        # outch1, outch2, outch3, outch_final = 4,8,8,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])


        # ## baseline 1
        # # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
        hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[-1])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[-2])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[-3])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        assert c==1
        return hypercorr_mix432.reshape(B_num_clips_1,hx*wx,hx1*wx1)

class HPNLearner2(nn.Module):
    ## for baselines 3, using 2 pyramid
    def __init__(self, inch):
        super(HPNLearner2, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])

        # ## baseline 1
        # # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
        hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[-1])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[-2])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[-3])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        # hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        # hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        assert c==1
        return hypercorr_mix43.reshape(B_num_clips_1,hx*wx,hx1*wx1)

class HPNLearner3(nn.Module):
    ## for baselines 4, only using one pyramid
    def __init__(self, inch):
        super(HPNLearner3, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])

        # ## baseline 1
        # # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
        hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[-1])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[-2])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[-3])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 #+ hypercorr_sqz3
        # hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        # hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        # hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        assert c==1
        return hypercorr_mix43.reshape(B_num_clips_1,hx*wx,hx1*wx1)

class HPNLearner_topk1(nn.Module):
    def __init__(self, inch):
        super(HPNLearner_topk1, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d_half(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1
        # outch1, outch2, outch3, outch_final = 2,4,4,1
        # outch1, outch2, outch3, outch_final = 4,8,8,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])


        # ## baseline 1
        # # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    def interpolate_support_dims(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
        hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
        return hypercorr

    def forward(self, hypercorr_pyramid):

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        # hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        # hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        # hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        # hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        # hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        # hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        # B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        # assert c==1
        # return hypercorr_mix432.reshape(B_num_clips_1,hx*wx,hx1*wx1)
        return [hypercorr_sqz4, hypercorr_sqz3, hypercorr_sqz2]

class HPNLearner_topk2(nn.Module):
    def __init__(self, inch):
        super(HPNLearner_topk2, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d_half(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1
        # outch1, outch2, outch3, outch_final = 2,4,4,1
        # outch1, outch2, outch3, outch_final = 4,8,8,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])

        # ## new way for better trade-off between speed and speed-up
        # outch1, outch2, outch_final = 1,2,1
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2], [3, 3], [1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2], [3, 3], [1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2], [5, 3], [1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch2, [outch2, outch2], [3, 3], [1, 1])
        # self.encoder_layer3to2 = make_building_block(outch2, [outch2, outch_final], [3, 3], [1, 1])

        # ## baseline 1
        # # Squeezing building blocks
        self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    # def interpolate_support_dims(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
    #     return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz_s, ch,  hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        # o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    # def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
    #     hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
    #     hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
    #     return hypercorr

    def forward(self, hypercorr_pyramid):
        ## atten shape: bsz_s,inch,hx,wx

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        # B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        # assert c==1
        return hypercorr_mix432

class HPNLearner_topk2_3d(nn.Module):
    def __init__(self, inch):
        super(HPNLearner_topk2_3d, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv3d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        # outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1
        # outch1, outch2, outch3, outch_final = 2,4,4,1
        # outch1, outch2, outch3, outch_final = 4,8,8,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])

        # ## new way for better trade-off between speed and speed-up
        outch1, outch2, outch_final = 1,2,1
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2], [3, 3], [1, 1])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2], [3, 3], [1, 1])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2], [5, 3], [1, 1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch2, [outch2, outch2], [3, 3], [1, 1])
        self.encoder_layer3to2 = make_building_block(outch2, [outch2, outch_final], [3, 3], [1, 1])

        # ## baseline 1
        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    # def interpolate_support_dims(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
    #     return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz, ch, s, hb, wb = hypercorr.size()
        hypercorr=hypercorr.reshape(bsz*ch,s,hb,wb)
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        # o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr.reshape(bsz,ch,s,spatial_size[0],spatial_size[1])

    # def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
    #     hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
    #     hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
    #     return hypercorr

    def forward(self, hypercorr_pyramid):
        ## atten shape: bsz_s,inch,hx,wx

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        # B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        # assert c==1
        return hypercorr_mix432

class HPNLearner_topk2_only_sar(nn.Module):
    def __init__(self, inch):
        super(HPNLearner_topk2_only_sar, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d_half(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        # outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1
        # outch1, outch2, outch3, outch_final = 2,4,4,1
        # outch1, outch2, outch3, outch_final = 4,8,8,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])

        # ## new way for better trade-off between speed and speed-up
        outch1, outch2, outch_final = 1,1,1
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2], [3, 3], [1, 1])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2], [3, 3], [1, 1])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2], [5, 3], [1, 1])

        # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch2, [outch2, outch2], [3, 3], [1, 1])
        # self.encoder_layer3to2 = make_building_block(outch2, [outch2, outch_final], [3, 3], [1, 1])

        # ## baseline 1
        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    # def interpolate_support_dims(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
    #     return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz_s, ch,  hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        # o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    # def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
    #     hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
    #     hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
    #     return hypercorr

    def forward(self, hypercorr_pyramid):
        ## atten shape: bsz_s,inch,hx,wx

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        # hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        # hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        # B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        # assert c==1
        return hypercorr_mix432

class HPNLearner_topk2_only_cfm(nn.Module):
    def __init__(self, inch):
        super(HPNLearner_topk2_only_cfm, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d_half(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        # outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1
        # outch1, outch2, outch3, outch_final = 2,4,4,1
        # outch1, outch2, outch3, outch_final = 4,8,8,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])

        # ## new way for better trade-off between speed and speed-up
        outch1, outch2, outch_final = 1,1,1
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2], [3, 3], [1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2], [3, 3], [1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2], [5, 3], [1, 1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch2, [outch2, outch2], [3, 3], [1, 1])
        self.encoder_layer3to2 = make_building_block(outch2, [outch2, outch_final], [3, 3], [1, 1])

        # ## baseline 1
        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    # def interpolate_support_dims(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
    #     return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz_s, ch,  hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        # o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    # def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
    #     hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
    #     hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
    #     return hypercorr

    def forward(self, hypercorr_pyramid):
        ## atten shape: bsz_s,inch,hx,wx

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = (hypercorr_pyramid[0])
        hypercorr_sqz3 = (hypercorr_pyramid[1])
        hypercorr_sqz2 = (hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43 + hypercorr_sqz2
        hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        # B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        # assert c==1
        return hypercorr_mix432

class HPNLearner_topk2_L1(nn.Module):
    def __init__(self, inch):
        super(HPNLearner_topk2_L1, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d_half(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        # outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1
        # outch1, outch2, outch3, outch_final = 2,4,4,1
        # outch1, outch2, outch3, outch_final = 4,8,8,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])

        # ## new way for better trade-off between speed and speed-up
        outch1, outch2, outch_final = 1,1,1
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2], [3, 3], [1, 1])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2], [3, 3], [1, 1])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2], [5, 3], [1, 1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch2, [outch2, outch2], [3, 3], [1, 1])
        self.encoder_layer3to2 = make_building_block(outch2, [outch2, outch_final], [3, 3], [1, 1])

        # ## baseline 1
        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    # def interpolate_support_dims(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
    #     return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz_s, ch,  hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        # o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    # def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
    #     hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
    #     hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
    #     return hypercorr

    def forward(self, hypercorr_pyramid):
        ## atten shape: bsz_s,inch,hx,wx

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 
        # hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43
        # hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        # B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        # assert c==1
        return hypercorr_mix432

class HPNLearner_topk2_L2(nn.Module):
    def __init__(self, inch):
        super(HPNLearner_topk2_L2, self).__init__()

        def make_building_block(in_channel, out_channels, kernel_sizes, spt_strides, group=1):
            assert len(out_channels) == len(kernel_sizes) == len(spt_strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, spt_strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(CenterPivotConv4d_half(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        # outch1, outch2, outch3 = 16, 64, 128
        # outch1, outch2, outch3, outch_final = 1,1,1,1
        # outch1, outch2, outch3, outch_final = 1,2,2,1
        # outch1, outch2, outch3, outch_final = 2,4,4,1
        # outch1, outch2, outch3, outch_final = 4,8,8,1

        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2, outch3], [5, 3, 3], [1, 1, 1])
        # self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2, outch3], [5, 5, 3], [1, 1, 1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [outch3, outch3, outch3], [3, 3, 3], [1, 1, 1])
        # self.encoder_layer3to2 = make_building_block(outch3, [outch3, outch3, outch_final], [3, 3, 3], [1, 1, 1])

        # ## new way for better trade-off between speed and speed-up
        outch1, outch2, outch_final = 1,1,1
        self.encoder_layer4 = make_building_block(inch[0], [outch1, outch2], [3, 3], [1, 1])
        self.encoder_layer3 = make_building_block(inch[1], [outch1, outch2], [3, 3], [1, 1])
        self.encoder_layer2 = make_building_block(inch[2], [outch1, outch2], [5, 3], [1, 1])

        # Mixing building blocks
        self.encoder_layer4to3 = make_building_block(outch2, [outch2, outch2], [3, 3], [1, 1])
        self.encoder_layer3to2 = make_building_block(outch2, [outch2, outch_final], [3, 3], [1, 1])

        # ## baseline 1
        # # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [3], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [5], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [5], [1])

        # # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [3], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [3], [1])

        ## baseline 2
        # Squeezing building blocks
        # self.encoder_layer4 = make_building_block(inch[0], [1], [1], [1])
        # self.encoder_layer3 = make_building_block(inch[1], [1], [1], [1])
        # self.encoder_layer2 = make_building_block(inch[2], [1], [1], [1])

        # # Mixing building blocks
        # self.encoder_layer4to3 = make_building_block(outch3, [1], [1], [1])
        # self.encoder_layer3to2 = make_building_block(outch3, [1], [1], [1])


        # Decoder layers
        # self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU())

        # self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
        #                               nn.ReLU(),
        #                               nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

    # def interpolate_support_dims(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     hypercorr = hypercorr.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     hypercorr = hypercorr.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
    #     return hypercorr

    def interpolate_support_dims2(self, hypercorr, spatial_size=None):
        bsz_s, ch,  hb, wb = hypercorr.size()
        # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
        hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
        # o_hb, o_wb = spatial_size
        # hypercorr = hypercorr.view(bsz, ha, wa, ch, o_hb, o_wb).permute(0, 3, 1, 2, 4, 5).contiguous()
        return hypercorr

    # def interpolate_support_dims2_fast(self, hypercorr, spatial_size=None):
    #     bsz, ch, ha, wa, hb, wb = hypercorr.size()
    #     # hypercorr = hypercorr.permute(0, 2, 3, 1, 4, 5).contiguous().view(bsz * ha * wa, ch, hb, wb)
    #     hypercorr = hypercorr.transpose(1,3).contiguous().view(bsz * wa * ha, ch, hb, wb)
    #     hypercorr = F.interpolate(hypercorr, spatial_size, mode='bilinear', align_corners=True)
    #     o_hb, o_wb = spatial_size
    #     # hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).permute(0, 3, 2, 1, 4, 5).contiguous()
    #     hypercorr = hypercorr.view(bsz, wa, ha, ch, o_hb, o_wb).transpose(1,3).contiguous()
    #     return hypercorr

    def forward(self, hypercorr_pyramid):
        ## atten shape: bsz_s,inch,hx,wx

        # Encode hypercorrelations from each layer (Squeezing building blocks)
        hypercorr_sqz4 = self.encoder_layer4(hypercorr_pyramid[0])
        hypercorr_sqz3 = self.encoder_layer3(hypercorr_pyramid[1])
        hypercorr_sqz2 = self.encoder_layer2(hypercorr_pyramid[2])
        # print(hypercorr_sqz4.shape, hypercorr_sqz3.shape, hypercorr_sqz2.shape)

        # Propagate encoded 4D-tensor (Mixing building blocks)
        # hypercorr_sqz4 = self.interpolate_support_dims(hypercorr_sqz4, hypercorr_sqz3.size()[-4:-2])
        hypercorr_sqz4 = self.interpolate_support_dims2(hypercorr_sqz4, hypercorr_sqz3.size()[-2:])
        hypercorr_mix43 = hypercorr_sqz4 + hypercorr_sqz3
        hypercorr_mix43 = self.encoder_layer4to3(hypercorr_mix43)

        # hypercorr_mix43 = self.interpolate_support_dims(hypercorr_mix43, hypercorr_sqz2.size()[-4:-2])
        hypercorr_mix43 = self.interpolate_support_dims2(hypercorr_mix43, hypercorr_sqz2.size()[-2:])
        hypercorr_mix432 = hypercorr_mix43 
        # hypercorr_mix432 = self.encoder_layer3to2(hypercorr_mix432)

        # bsz, ch, ha, wa, hb, wb = hypercorr_mix432.size()
        # hypercorr_encoded = hypercorr_mix432.view(bsz, ch, ha, wa, -1).mean(dim=-1)

        # Decode the encoded 4D-tensor
        # hypercorr_decoded = self.decoder1(hypercorr_encoded)
        # upsample_size = (hypercorr_decoded.size(-1) * 2,) * 2
        # hypercorr_decoded = F.interpolate(hypercorr_decoded, upsample_size, mode='bilinear', align_corners=True)
        # logit_mask = self.decoder2(hypercorr_decoded)

        # return logit_mask
        # B_num_clips_1,c,hx,wx,hx1,wx1=hypercorr_pyramid[-3].shape
        # assert c==1
        return hypercorr_mix432



class hypercorre(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
    num_feats: number of features being used
    """

    def __init__(self,
                 stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4):
        super().__init__()
        self.stack_id=stack_id
        self.dim=dim
        self.num_qkv=2
        # self.qkv_bias=qkv_bias
        # self.qkv0 = nn.Linear(dim[0], dim[0] * self.num_qkv, bias=qkv_bias)
        self.qkv1 = nn.Linear(dim[1], dim[1] * self.num_qkv, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim[2], dim[2] * self.num_qkv, bias=qkv_bias)
        self.qkv3 = nn.Linear(dim[3], dim[3] * self.num_qkv, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim , bias=qkv_bias)

        self.hpn=HPNLearner([1,1,1])
        # self.hpn=HPNLearner2([1,1,1])
        # self.hpn=HPNLearner3([1,1,1])
        
    def forward(self, query_frame, supp_frame):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        Args:
            
        """
        start_time=time.time()
        query_qkv_all=[]
        query_shape_all=[]
        for ii, query in enumerate(query_frame):
            B,num_ref_clips,cx,hx,wx=query.shape
            if ii==0:
                # query_qkv=self.qkv0(query.permute(0,1,3,4,2))
                query_qkv_all.append(None)
                query_shape_all.append([None,None])
                continue
            elif ii==1:
                query_qkv=self.qkv1(query.permute(0,1,3,4,2))
            elif ii==2:
                query_qkv=self.qkv2(query.permute(0,1,3,4,2))
            elif ii==3:
                query_qkv=self.qkv3(query.permute(0,1,3,4,2))
            query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,hx*wx,self.num_qkv,cx))
            query_shape_all.append([hx, wx])

        supp_qkv_all=[]
        atten_all=[]
        for ii, supp in enumerate(supp_frame):
            B,num_ref_clips,cx,hx,wx=supp.shape
            if ii==0:
                # supp_qkv=self.qkv0(supp.permute(0,1,3,4,2))
                supp_qkv_all.append(None)
                continue
            elif ii==1:
                supp_qkv=self.qkv1(supp.permute(0,1,3,4,2))
            elif ii==2:
                supp_qkv=self.qkv2(supp.permute(0,1,3,4,2))
            elif ii==3:
                supp_qkv=self.qkv3(supp.permute(0,1,3,4,2))
            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,self.num_qkv,cx))    

            if ii>0:
                atten=torch.matmul(supp_qkv_all[ii][:,:,:,0,:], query_qkv_all[ii][:,:,:,1,:].transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(hx*wx)
                atten=atten.reshape(B*atten.shape[1],1,hx,wx,query_shape_all[ii][0],query_shape_all[ii][1])    ## (B*(num_clips-1))*1*hx*wx*hx*wx
                atten_all.append(atten)
        #         print(atten.shape) # [60, 60, 30, 30],[30, 30, 30, 30],[15, 15, 15, 15]
        # print(len(atten_all))
        start_time1=time.time()
        atten_new=self.hpn(atten_all)
        start_time2=time.time()
        B,num_ref_clips,_,_,_=query_frame[0].shape
        atten_new=atten_new.reshape(B,num_ref_clips,atten_new.shape[-2],atten_new.shape[-1])
        start_time3=time.time()
        # print("here: ", start_time1-start_time, start_time2-start_time1, start_time3-start_time2)
        return atten_new

class hypercorre_topk1(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
    num_feats: number of features being used
    """

    def __init__(self,
                 stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4):
        super().__init__()
        self.stack_id=stack_id
        self.dim=dim
        # self.num_qkv=2
        # self.qkv_bias=qkv_bias
        # self.qkv0 = nn.Linear(dim[0], dim[0] * self.num_qkv, bias=qkv_bias)
        # self.qkv1 = nn.Linear(dim[1], dim[1] * self.num_qkv, bias=qkv_bias)
        # self.qkv2 = nn.Linear(dim[2], dim[2] * self.num_qkv, bias=qkv_bias)
        # self.qkv3 = nn.Linear(dim[3], dim[3] * self.num_qkv, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.q1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.q2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.q3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.k1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.k2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.k3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.hpn=HPNLearner_topk1([1,1,1])
        # self.hpn=HPNLearner2([1,1,1])
        # self.hpn=HPNLearner3([1,1,1])
        
    def forward(self, query_frame, supp_frame):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        Args:
            
        """
        start_time=time.time()
        query_frame=query_frame[::-1]
        supp_frame=supp_frame[::-1]
        query_qkv_all=[]
        query_shape_all=[]
        for ii, query in enumerate(query_frame):
            B,num_ref_clips,cx,hy,wy=query.shape
            if ii==0:
                query_qkv=self.k3(query.permute(0,1,3,4,2))
            elif ii==1:
                query_qkv=self.k2(query.permute(0,1,3,4,2))
            elif ii==2:
                query_qkv=self.k1(query.permute(0,1,3,4,2))
            elif ii==3:
                query_qkv_all.append(None)
                query_shape_all.append([None,None])
                continue
            query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,hy*wy,cx))       ## B,num_ref_clips,hy,wy,cx
            query_shape_all.append([hy, wy])

        supp_qkv_all=[]
        supp_shape_all=[]
        for ii, supp in enumerate(supp_frame):
            B,num_ref_clips,cx,hx,wx=supp.shape
            if ii==0:
                supp_qkv=self.q3(supp.permute(0,1,3,4,2))    
            elif ii==1:
                supp_qkv=self.q2(supp.permute(0,1,3,4,2))
            elif ii==2:
                supp_qkv=self.q1(supp.permute(0,1,3,4,2))
            elif ii==3:
                supp_qkv_all.append(None)
                continue
            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,cx))    ## B,num_ref_clips,hx*wx,cx
            supp_shape_all.append([hx,wx])

        loca_selection=[]
        indices_sele=[]
        k_top=5
        threh=0.8
        # query_qkv_all=query_qkv_all[::-1]
        # query_shape_all=query_shape_all[::-1]
        # supp_qkv_all=supp_qkv_all[::-1]
        # supp_shape_all=supp_shape_all[::-1]
        atten_all=[]
        s_all=[]

        B=supp_qkv_all[0].shape[0]
        q_num_ref=query_qkv_all[0].shape[1]
        for ii in range(0,len(supp_frame)-1):
            hy,wy=query_shape_all[ii]
            
            hx,wx=supp_shape_all[ii]
            if ii==0:
                atten=torch.matmul(supp_qkv_all[ii], query_qkv_all[ii].transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(hy*wy)
                atten_fullmatrix=atten
            else:
                cx=query_qkv_all[ii].shape[-1]
                query_selected=query_qkv_all[ii]  ## B,(num_clips-1),hy*wy,c
                assert query_selected.shape[:-1]==loca_selection[ii-1].shape     
                
                num_selection=torch.unique((loca_selection[ii-1]).sum(2))
                assert num_selection.shape[0]==1 and num_selection.dim()==1
                # query_selected=torch.masked_select(querry_selected, loca_selection[ii-2].unsqueeze(-1))
                query_selected=query_selected[loca_selection[ii-1]]
                query_selected=query_selected.reshape(B, q_num_ref, num_selection[0],cx)     ##  B*(num_clips-1)*s*c

                atten=torch.matmul(supp_qkv_all[ii], query_selected.transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(s)
                atten_fullmatrix=-100*torch.ones(B,q_num_ref,hx*wx,hy*wy).cuda()
                indices=indices_sele[ii-1]
                assert indices.shape[-1]==num_selection[0]
                indices=indices.unsqueeze(2).expand(B,q_num_ref,hx*wx,num_selection[0])    ## B*(num_clips-1)*(hx*wx)*(s)
                atten_fullmatrix=atten_fullmatrix.scatter(3,indices,atten)
            if ii<len(supp_frame)-2:
                # atten_temp=atten.reshape(B*atten.shape[1],hx*wx,query_shape_all[ii][0],query_shape_all[ii][1])
                # atten_topk=F.interpolate()
                atten_topk=torch.topk(atten_fullmatrix,k_top,dim=2)[0]    # B*(num_clips-1)*(k)*(hy*wy)
                atten_topk=atten_topk.sum(2)   # B*(num_clips-1)*(hy*wy)
                # atten_kthvalue=torch.kthvalue(atten_topk,atten_topk.shape[-1]*threh,dim=2)[0]   # 
                # topk_mask=atten_topk>atten_kthvalue   # B*(num_clips-1)*(hy*wy)
                # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], hy, wy)  # B*(num_clips-1)*hy*wy
                
                hy_next, wy_next=query_shape_all[ii+1]
                if hy!=hy_next or wy!=wy_next:
                    atten_topk=atten_topk.reshape(B,q_num_ref,hy,wy)    # B*(num_clips-1)*hy*wy
                    atten_topk=F.interpolate(atten_topk, (hy_next, wy_next), mode='bilinear', align_corners=False)    # B*(num_clips-1)*hy_next*wy_next
                    atten_topk=atten_topk.reshape(B, q_num_ref, hy_next*wy_next)   # B*(num_clips-1)*(hy_next*wy_next)

                indices=torch.topk(atten_topk,int(atten_topk.shape[-1]*threh**(ii+1)),dim=2)[1]    # # B*(num_clips-1)*s
                topk_mask=torch.zeros_like(atten_topk)
                topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy_next*wy_next)
                # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], hy_next, wy_next)
                # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], query_shape_all[ii][0], query_shape_all[ii][1])  # B*(num_clips-1)*hy*wy
                # loca_selection[ii-1]=F.interpolate(topk_mask, query_shape_all[ii], mode='nearest', align_corners=False)>0.5
                loca_selection.append(topk_mask>0.5)
                indices_sele.append(indices)
            # else:
                # atten=torch.zeros()

            s=atten.shape[3]
            atten=atten.permute(0,1,3,2).reshape(B*q_num_ref*s,hx,wx)   ## (B*(num_clips-1))*hx*wx*s
            atten_all.append(atten.unsqueeze(1))
            s_all.append(s)
        #         print(atten.shape) # [60, 60, 30, 30],[30, 30, 30, 30],[15, 15, 15, 15]
        # print(len(atten_all))
        start_time1=time.time()
        atten_all=self.hpn(atten_all)
        start_time2=time.time()
        # B,num_ref_clips,_,_,_=query_frame[0].shape
        # atten_new=atten_new.reshape(B,num_ref_clips,atten_new.shape[-2],atten_new.shape[-1])
        atten_all=[atten_one.squeeze(1).reshape(B,q_num_ref,s_all[i],supp_shape_all[i][0]*supp_shape_all[i][1]).permute(0,1,3,2) for i,atten_one in enumerate(atten_all)]
        start_time3=time.time()
        # print("here: ", start_time1-start_time, start_time2-start_time1, start_time3-start_time2)
        return atten_all, loca_selection, s_all     # len(atten_all)=3, len(loca_selection)=2
    
class hypercorre_topk2(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used

    Args:
    num_feats: number of features being used

    """

    def __init__(self,
                 stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4):
        super().__init__()
        self.stack_id=stack_id
        self.dim=dim
        # self.num_qkv=2
        # self.qkv_bias=qkv_bias
        # self.qkv0 = nn.Linear(dim[0], dim[0] * self.num_qkv, bias=qkv_bias)
        # self.qkv1 = nn.Linear(dim[1], dim[1] * self.num_qkv, bias=qkv_bias)
        # self.qkv2 = nn.Linear(dim[2], dim[2] * self.num_qkv, bias=qkv_bias)
        # self.qkv3 = nn.Linear(dim[3], dim[3] * self.num_qkv, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.q1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.q2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.q3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.k1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.k2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.k3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.hpn=HPNLearner_topk2([1,1,1])

        ## for various ablation studies
        # self.hpn=HPNLearner_topk2_only_sar([1,1,1])
        # self.hpn=HPNLearner_topk2_only_cfm([1,1,1])
        # self.hpn=HPNLearner2([1,1,1])
        # self.hpn=HPNLearner3([1,1,1])
        # self.hpn=HPNLearner_topk2_L1([1,1,1])
        # self.hpn=HPNLearner_topk2_L2([1,1,1])
        self.use_3dconv=False
        # self.use_3dconv=True
        # if self.use_3dconv:
        #     self.hpn=HPNLearner_topk2_3d([1,1,1])    ## ablation study of 3d conv needs a full shape of affinity
        
    def forward(self, query_frame, supp_frame):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        Args:
            
        """
        start_time=time.time()
        query_frame=query_frame[::-1]
        supp_frame=supp_frame[::-1]
        query_qkv_all=[]
        query_shape_all=[]
        for ii, query in enumerate(query_frame):
            B,num_ref_clips,cx,hy,wy=query.shape
            if ii==0:
                query_qkv=self.k3(query.permute(0,1,3,4,2))
            elif ii==1:
                query_qkv=self.k2(query.permute(0,1,3,4,2))
            elif ii==2:
                query_qkv=self.k1(query.permute(0,1,3,4,2))
            elif ii==3:
                ## skip h/4*w/4 feature because it is too big
                query_qkv_all.append(None)
                query_shape_all.append([None,None])
                continue
            query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,hy*wy,cx))       ## B,num_ref_clips,hy*wy,cx
            query_shape_all.append([hy, wy])

        supp_qkv_all=[]
        supp_shape_all=[]
        for ii, supp in enumerate(supp_frame):
            B,num_ref_clips,cx,hx,wx=supp.shape
            if ii==0:
                supp_qkv=self.q3(supp.permute(0,1,3,4,2))    
            elif ii==1:
                supp_qkv=self.q2(supp.permute(0,1,3,4,2))
            elif ii==2:
                supp_qkv=self.q1(supp.permute(0,1,3,4,2))
            elif ii==3:
                supp_qkv_all.append(None)
                supp_shape_all.append([None,None])
                continue
            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,cx))    ## B,num_ref_clips,hx*wx,cx
            supp_shape_all.append([hx,wx])

        loca_selection=[]
        indices_sele=[]
        k_top=5
        threh=0.5
        # print(threh,k_top)

        # query_qkv_all=query_qkv_all[::-1]
        # query_shape_all=query_shape_all[::-1]
        # supp_qkv_all=supp_qkv_all[::-1]
        # supp_shape_all=supp_shape_all[::-1]
        atten_all=[]
        s_all=[]

        B=supp_qkv_all[0].shape[0]
        q_num_ref=query_qkv_all[0].shape[1]
        # print(query_shape_all)
        for ii in range(0,len(supp_frame)-1):
            hy,wy=query_shape_all[ii]
            hx,wx=supp_shape_all[ii]
            if ii==0:
                atten=torch.matmul(supp_qkv_all[ii], query_qkv_all[ii].transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(hy*wy)
                atten_fullmatrix=atten
            else:
                cx=query_qkv_all[ii].shape[-1]
                query_selected=query_qkv_all[ii]  ## B*(num_clips-1)*(hy*wy)*c
                assert query_selected.shape[:-1]==loca_selection[ii-1].shape     
                
                num_selection=torch.unique((loca_selection[ii-1]).sum(2))
                assert num_selection.shape[0]==1 and num_selection.dim()==1
                # query_selected=torch.masked_select(querry_selected, loca_selection[ii-2].unsqueeze(-1))
                query_selected=query_selected[loca_selection[ii-1]>0.5].reshape(B, q_num_ref, int(num_selection[0]),cx)     ##  B*(num_clips-1)*s*c

                atten=torch.matmul(supp_qkv_all[ii], query_selected.transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(s)
                # atten_fullmatrix=-100*torch.ones(B,q_num_ref,atten.shape[2],(query_shape_all[ii][0]*query_shape_all[ii][1])).cuda()
                # indices=indices_sele[ii-1]
                # assert indices.shape[-1]==num_selection[0]
                # indices=indices.unsqueeze(2).expand(B,q_num_ref,atten.shape[2],indices.shape[-1])    ## B*(num_clips-1)*(hx*wx)*(s)
                # atten_fullmatrix=atten_fullmatrix.scatter(3,indices,atten)
            if ii==0:
                # atten_temp=atten.reshape(B*atten.shape[1],hx*wx,query_shape_all[ii][0],query_shape_all[ii][1])
                # atten_topk=F.interpolate()
                atten_topk=torch.topk(atten_fullmatrix,k_top,dim=2)[0]    # B*(num_clips-1)*(k)*(hy*wy)
                atten_topk=atten_topk.sum(2)   # B*(num_clips-1)*(hy*wy)
                # atten_kthvalue=torch.kthvalue(atten_topk,atten_topk.shape[-1]*threh,dim=2)[0]   # 
                # topk_mask=atten_topk>atten_kthvalue   # B*(num_clips-1)*(hy*wy)
                # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], hy, wy)  # B*(num_clips-1)*hy*wy
                # s=int(hy*wy*threh)
                # indices=torch.topk(atten_topk,s,dim=2)[1]    # B*(num_clips-1)*s
                # topk_mask=torch.zeros_like(atten_topk)
                # topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy*wy)
                # atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)

                # topk_mask=topk_mask.reshape(B, q_num_ref, hy, wy)
                
                hy_next, wy_next=query_shape_all[ii+1]

                if hy_next==hy and wy_next==wy:
                    s=int(hy*wy*threh)
                    indices=torch.topk(atten_topk,s,dim=2)[1]    # B*(num_clips-1)*s
                    topk_mask=torch.zeros_like(atten_topk)
                    topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy*wy)
                    atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)

                else:   # hy_next!=hy or wy_next!=wy
                    atten=atten.reshape(B*q_num_ref,hx*wx,hy,wy)
                    atten=F.interpolate(atten, (hy_next, wy_next), mode='bilinear', align_corners=False).reshape(B,q_num_ref,hx*wx,hy_next*wy_next)

                    atten_topk=atten_topk.reshape(B,q_num_ref,hy,wy)    # B*(num_clips-1)*hy*wy
                    atten_topk=F.interpolate(atten_topk, (hy_next, wy_next), mode='bilinear', align_corners=False)    # B*(num_clips-1)*hy_next*wy_next
                    atten_topk=atten_topk.reshape(B, q_num_ref, hy_next*wy_next)   # B*(num_clips-1)*(hy_next*wy_next)

                    s=int(hy_next*wy_next*threh)
                    indices=torch.topk(atten_topk,s,dim=2)[1]    # # B*(num_clips-1)*s
                    topk_mask=torch.zeros_like(atten_topk)
                    topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy_next*wy_next)

                    atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)
                    # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], hy_next, wy_next)
                # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], query_shape_all[ii][0], query_shape_all[ii][1])  # B*(num_clips-1)*hy*wy
                # loca_selection[ii-1]=F.interpolate(topk_mask, query_shape_all[ii], mode='nearest', align_corners=False)>0.5
                loca_selection.append(topk_mask)
                # indices_sele[0]=indices
            elif ii<=len(supp_frame)-3:
                loca_selection.append(loca_selection[-1])

            s=atten.shape[3]
            # print("here: ", atten.shape, atten.max(), atten.min())
            if self.use_3dconv:   
                # ablation study of 3d_conv needs a full shape of atten 
                atten=atten.permute(0,1,3,2).reshape(B*q_num_ref,s,hx,wx)   ## (B*(num_clips-1))*s*hx*wx
            else:
                atten=atten.permute(0,1,3,2).reshape(B*q_num_ref*s,hx,wx)   ## (B*(num_clips-1)*s)*hx*wx
            atten_all.append(atten.unsqueeze(1))
            s_all.append(s)
        #         print(atten.shape) # [60, 60, 30, 30],[30, 30, 30, 30],[15, 15, 15, 15]
        # print(len(atten_all))
        start_time1=time.time()
        atten_all=self.hpn(atten_all)
        start_time2=time.time()
        # B,num_ref_clips,_,_,_=query_frame[0].shape
        # atten_new=atten_new.reshape(B,num_ref_clips,atten_new.shape[-2],atten_new.shape[-1])
        # atten_all=[atten_one.squeeze(1).reshape(B,q_num_ref,s_all[i],supp_shape_all.shape[0],supp_shape_all.shape[1]).permute(0,1,3,4,2) for i,atten_one in enumerate(atten_all)]
        if self.use_3dconv:
            # ablation study of 3d conv uses a full shape of atten, so reshaping should adjust accordingly
            atten_all=atten_all.squeeze(1).reshape(B,q_num_ref,s_all[-1],supp_shape_all[2][0]*supp_shape_all[2][1]).permute(0,1,3,2)   # B*(num_clips-1)*(hx*wx)*s
        else:
            atten_all=atten_all.squeeze(1).reshape(B,q_num_ref,s_all[-1],supp_shape_all[2][0]*supp_shape_all[2][1]).permute(0,1,3,2)   # B*(num_clips-1)*(hx*wx)*s
        # print(atten_all.shape)
        start_time3=time.time()
        # print("here: ", start_time1-start_time, start_time2-start_time1, start_time3-start_time2)
        return atten_all, loca_selection[-1]>0.5

class multi_scale_atten(nn.Module):
    """ top-k2: same selections for each reference image so that attention decoder can be used

    Args:
    num_feats: number of features being used

    """

    def __init__(self,
                 stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4):
        super().__init__()
        self.stack_id=stack_id
        self.dim=dim
        # self.num_qkv=2
        # self.qkv_bias=qkv_bias
        # self.qkv0 = nn.Linear(dim[0], dim[0] * self.num_qkv, bias=qkv_bias)
        # self.qkv1 = nn.Linear(dim[1], dim[1] * self.num_qkv, bias=qkv_bias)
        # self.qkv2 = nn.Linear(dim[2], dim[2] * self.num_qkv, bias=qkv_bias)
        # self.qkv3 = nn.Linear(dim[3], dim[3] * self.num_qkv, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim , bias=qkv_bias)
        self.q1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.q2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.q3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        self.k1 = nn.Linear(dim[1], dim[1], bias=qkv_bias)
        self.k2 = nn.Linear(dim[2], dim[2], bias=qkv_bias)
        self.k3 = nn.Linear(dim[3], dim[3], bias=qkv_bias)

        # self.hpn=HPNLearner_topk2([1,1,1])
        
    def forward(self, query_frame, supp_frame):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        Args:
            
        """
        start_time=time.time()
        query_frame=query_frame[::-1]
        supp_frame=supp_frame[::-1]
        query_qkv_all=[]
        query_shape_all=[]
        for ii, query in enumerate(query_frame):
            B,num_ref_clips,cx,hy,wy=query.shape
            if ii==0:
                query_qkv=self.k3(query.permute(0,1,3,4,2))
            elif ii==1:
                query_qkv=self.k2(query.permute(0,1,3,4,2))
            elif ii==2:
                query_qkv=self.k1(query.permute(0,1,3,4,2))
            elif ii==3:
                ## skip h/4*w/4 feature because it is too big
                query_qkv_all.append(None)
                query_shape_all.append([None,None])
                continue
            query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,hy*wy,cx))       ## B,num_ref_clips,hy*wy,cx
            query_shape_all.append([hy, wy])

        supp_qkv_all=[]
        supp_shape_all=[]
        for ii, supp in enumerate(supp_frame):
            B,num_ref_clips,cx,hx,wx=supp.shape
            if ii==0:
                supp_qkv=self.q3(supp.permute(0,1,3,4,2))    
            elif ii==1:
                supp_qkv=self.q2(supp.permute(0,1,3,4,2))
            elif ii==2:
                supp_qkv=self.q1(supp.permute(0,1,3,4,2))
            elif ii==3:
                supp_qkv_all.append(None)
                supp_shape_all.append([None,None])
                continue
            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,cx))    ## B,num_ref_clips,hx*wx,cx
            supp_shape_all.append([hx,wx])

        loca_selection=[]
        indices_sele=[]
        k_top=5
        threh=0.5
        # print(threh,k_top)

        # query_qkv_all=query_qkv_all[::-1]
        # query_shape_all=query_shape_all[::-1]
        # supp_qkv_all=supp_qkv_all[::-1]
        # supp_shape_all=supp_shape_all[::-1]
        atten_all=[]
        s_all=[]

        B=supp_qkv_all[0].shape[0]
        q_num_ref=query_qkv_all[0].shape[1]
        # print(query_shape_all)
        for ii in range(0,len(supp_frame)-1):
            hy,wy=query_shape_all[ii]
            hx,wx=supp_shape_all[ii]
            # if ii==0:
            atten=torch.matmul(supp_qkv_all[ii], query_qkv_all[ii].transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(hy*wy)
            atten_all.append(atten)
            #     atten_fullmatrix=atten
            # else:
            #     cx=query_qkv_all[ii].shape[-1]
            #     query_selected=query_qkv_all[ii]  ## B*(num_clips-1)*(hy*wy)*c
            #     assert query_selected.shape[:-1]==loca_selection[ii-1].shape     
                
            #     num_selection=torch.unique((loca_selection[ii-1]).sum(2))
            #     assert num_selection.shape[0]==1 and num_selection.dim()==1
            #     # query_selected=torch.masked_select(querry_selected, loca_selection[ii-2].unsqueeze(-1))
            #     query_selected=query_selected[loca_selection[ii-1]>0.5].reshape(B, q_num_ref, int(num_selection[0]),cx)     ##  B*(num_clips-1)*s*c

            #     atten=torch.matmul(supp_qkv_all[ii], query_selected.transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(s)
                # atten_fullmatrix=-100*torch.ones(B,q_num_ref,atten.shape[2],(query_shape_all[ii][0]*query_shape_all[ii][1])).cuda()
                # indices=indices_sele[ii-1]
                # assert indices.shape[-1]==num_selection[0]
                # indices=indices.unsqueeze(2).expand(B,q_num_ref,atten.shape[2],indices.shape[-1])    ## B*(num_clips-1)*(hx*wx)*(s)
                # atten_fullmatrix=atten_fullmatrix.scatter(3,indices,atten)
            # if ii==0:
            #     # atten_temp=atten.reshape(B*atten.shape[1],hx*wx,query_shape_all[ii][0],query_shape_all[ii][1])
            #     # atten_topk=F.interpolate()
            #     atten_topk=torch.topk(atten_fullmatrix,k_top,dim=2)[0]    # B*(num_clips-1)*(k)*(hy*wy)
            #     atten_topk=atten_topk.sum(2)   # B*(num_clips-1)*(hy*wy)
            #     # atten_kthvalue=torch.kthvalue(atten_topk,atten_topk.shape[-1]*threh,dim=2)[0]   # 
            #     # topk_mask=atten_topk>atten_kthvalue   # B*(num_clips-1)*(hy*wy)
            #     # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], hy, wy)  # B*(num_clips-1)*hy*wy
            #     # s=int(hy*wy*threh)
            #     # indices=torch.topk(atten_topk,s,dim=2)[1]    # B*(num_clips-1)*s
            #     # topk_mask=torch.zeros_like(atten_topk)
            #     # topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy*wy)
            #     # atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)

            #     # topk_mask=topk_mask.reshape(B, q_num_ref, hy, wy)
                
            #     hy_next, wy_next=query_shape_all[ii+1]

            #     if hy_next==hy and wy_next==wy:
            #         s=int(hy*wy*threh)
            #         indices=torch.topk(atten_topk,s,dim=2)[1]    # B*(num_clips-1)*s
            #         topk_mask=torch.zeros_like(atten_topk)
            #         topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy*wy)
            #         atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)

            #     else:   # hy_next!=hy or wy_next!=wy
            #         atten=atten.reshape(B*q_num_ref,hx*wx,hy,wy)
            #         atten=F.interpolate(atten, (hy_next, wy_next), mode='bilinear', align_corners=False).reshape(B,q_num_ref,hx*wx,hy_next*wy_next)

            #         atten_topk=atten_topk.reshape(B,q_num_ref,hy,wy)    # B*(num_clips-1)*hy*wy
            #         atten_topk=F.interpolate(atten_topk, (hy_next, wy_next), mode='bilinear', align_corners=False)    # B*(num_clips-1)*hy_next*wy_next
            #         atten_topk=atten_topk.reshape(B, q_num_ref, hy_next*wy_next)   # B*(num_clips-1)*(hy_next*wy_next)

            #         s=int(hy_next*wy_next*threh)
            #         indices=torch.topk(atten_topk,s,dim=2)[1]    # # B*(num_clips-1)*s
            #         topk_mask=torch.zeros_like(atten_topk)
            #         topk_mask=topk_mask.scatter(2,indices,1)    # B*(num_clips-1)*(hy_next*wy_next)

            #         atten=atten[topk_mask.unsqueeze(2).expand_as(atten)>0.5].reshape(B,q_num_ref,hx*wx,s)
            #         # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], hy_next, wy_next)
            #     # topk_mask=topk_mask.reshape(B, topk_mask.shape[1], query_shape_all[ii][0], query_shape_all[ii][1])  # B*(num_clips-1)*hy*wy
            #     # loca_selection[ii-1]=F.interpolate(topk_mask, query_shape_all[ii], mode='nearest', align_corners=False)>0.5
            #     loca_selection.append(topk_mask)
            #     # indices_sele[0]=indices
            # elif ii<=len(supp_frame)-3:
            #     loca_selection.append(loca_selection[-1])

            # s=atten.shape[3]
            # # print("here: ", atten.shape, atten.max(), atten.min())
            # if self.use_3dconv:   
            #     # ablation study of 3d_conv needs a full shape of atten 
            #     atten=atten.permute(0,1,3,2).reshape(B*q_num_ref,s,hx,wx)   ## (B*(num_clips-1))*s*hx*wx
            # else:
            #     atten=atten.permute(0,1,3,2).reshape(B*q_num_ref*s,hx,wx)   ## (B*(num_clips-1)*s)*hx*wx
            # atten_all.append(atten.unsqueeze(1))
            # s_all.append(s)
        #         print(atten.shape) # [60, 60, 30, 30],[30, 30, 30, 30],[15, 15, 15, 15]
        # print(len(atten_all))
        # start_time1=time.time()
        # atten_all=self.hpn(atten_all)
        # start_time2=time.time()
        # # B,num_ref_clips,_,_,_=query_frame[0].shape
        # # atten_new=atten_new.reshape(B,num_ref_clips,atten_new.shape[-2],atten_new.shape[-1])
        # # atten_all=[atten_one.squeeze(1).reshape(B,q_num_ref,s_all[i],supp_shape_all.shape[0],supp_shape_all.shape[1]).permute(0,1,3,4,2) for i,atten_one in enumerate(atten_all)]
        # if self.use_3dconv:
        #     # ablation study of 3d conv uses a full shape of atten, so reshaping should adjust accordingly
        #     atten_all=atten_all.squeeze(1).reshape(B,q_num_ref,s_all[-1],supp_shape_all[2][0]*supp_shape_all[2][1]).permute(0,1,3,2)   # B*(num_clips-1)*(hx*wx)*s
        # else:
        #     atten_all=atten_all.squeeze(1).reshape(B,q_num_ref,s_all[-1],supp_shape_all[2][0]*supp_shape_all[2][1]).permute(0,1,3,2)   # B*(num_clips-1)*(hx*wx)*s
        # # print(atten_all.shape)
        # start_time3=time.time()
        # print("here: ", start_time1-start_time, start_time2-start_time1, start_time3-start_time2)
        return atten_all

class hypercorre2(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
    num_feats: number of features being used
    """

    def __init__(self,
                 stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4):
        super().__init__()
        self.stack_id=stack_id
        self.dim=dim
        self.num_qkv=2
        # self.qkv_bias=qkv_bias
        self.qkv0 = nn.Linear(dim[0], dim[0] * self.num_qkv, bias=qkv_bias)
        self.qkv1 = nn.Linear(dim[1], dim[1] * self.num_qkv, bias=qkv_bias)
        self.qkv2 = nn.Linear(dim[2], dim[2] * self.num_qkv, bias=qkv_bias)
        self.qkv3 = nn.Linear(dim[3], dim[3] * self.num_qkv, bias=qkv_bias)
        # self.q = nn.Linear(dim, dim , bias=qkv_bias)

        self.hpn=HPNLearner([1,1,1])
        # self.hpn=HPNLearner2([1,1,1])
        # self.hpn=HPNLearner3([1,1,1])
        
    def forward(self, query_frame, supp_frame):
        """ Forward function.
        query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
        supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
        Args:
            
        """
        start_time=time.time()
        query_qkv_all=[]
        query_shape_all=[]
        for ii, query in enumerate(query_frame):
            B,num_ref_clips,cx,hx,wx=query.shape
            if ii==0:
                query_qkv=self.qkv0(query.permute(0,1,3,4,2))
            elif ii==1:
                query_qkv=self.qkv1(query.permute(0,1,3,4,2))
            elif ii==2:
                query_qkv=self.qkv2(query.permute(0,1,3,4,2))
            elif ii==3:
                query_qkv=self.qkv3(query.permute(0,1,3,4,2))
            query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,hx*wx,self.num_qkv,cx))
            query_shape_all.append([hx, wx])

        supp_qkv_all=[]
        atten_all=[]
        for ii, supp in enumerate(supp_frame):
            B,num_ref_clips,cx,hx,wx=supp.shape
            if ii==0:
                supp_qkv=self.qkv0(supp.permute(0,1,3,4,2))
            elif ii==1:
                supp_qkv=self.qkv1(supp.permute(0,1,3,4,2))
            elif ii==2:
                supp_qkv=self.qkv2(supp.permute(0,1,3,4,2))
            elif ii==3:
                supp_qkv=self.qkv3(supp.permute(0,1,3,4,2))
            supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,self.num_qkv,cx))    

            if ii>0:
                atten=torch.matmul(supp_qkv_all[ii][:,:,:,0,:], query_qkv_all[ii][:,:,:,1,:].transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(hx*wx)
                atten=atten.reshape(B*atten.shape[1],1,hx,wx,query_shape_all[ii][0],query_shape_all[ii][1])    ## (B*(num_clips-1))*1*hx*wx*hx*wx
                atten_all.append(atten)
        #         print(atten.shape) # [60, 60, 30, 30],[30, 30, 30, 30],[15, 15, 15, 15]
        # print(len(atten_all))
        start_time1=time.time()
        atten_new=self.hpn(atten_all)
        start_time2=time.time()
        B,num_ref_clips,_,_,_=query_frame[0].shape
        atten_new=atten_new.reshape(B,num_ref_clips,atten_new.shape[-2],atten_new.shape[-1])
        start_time3=time.time()
        # print("here: ", start_time1-start_time, start_time2-start_time1, start_time3-start_time2)
        return atten_new
        


# class hypercorre2(nn.Module):
#     """ A basic Swin Transformer layer for one stage.

#     Args:
#     num_feats: number of features being used
#     """

#     def __init__(self,
#                  stack_id=None, dim=[64, 128, 320, 512], qkv_bias=True, num_feats=4):
#         super().__init__()
#         self.stack_id=stack_id
#         self.dim=dim
#         self.num_qkv=2
#         # self.qkv_bias=qkv_bias
#         # self.qkv0 = nn.Linear(dim[0], dim[0] * self.num_qkv, bias=qkv_bias)
#         # self.qkv1 = nn.Linear(dim[1], dim[1] * self.num_qkv, bias=qkv_bias)
#         # self.qkv2 = nn.Linear(dim[2], dim[2] * self.num_qkv, bias=qkv_bias)
#         # self.qkv3 = nn.Linear(dim[3], dim[3] * self.num_qkv, bias=qkv_bias)
#         # self.qkv0 = nn.Conv2d(dim[0], dim[0], (1,1), stride=(1,1), bias=qkv_bias, padding=(0,0))
#         # self.q = nn.Linear(dim, dim , bias=qkv_bias)

#         self.hpn=HPNLearner([1,1,1])
        
#     def forward(self, query_frame, supp_frame):
#         """ Forward function.
#         query_frame: [B*(num_clips-1)*c*h/4*w/4, B*(num_clips-1)*c*h/8*w/8, B*(num_clips-1)*c*h/16*w/16, B*(num_clips-1)*c*h/32*w/32]
#         supp_frame: [B*1*c*h/4*w/4, B*1*c*h/8*w/8, B*1*c*h/16*w/16, B*1*c*h/32*w/32]
#         Args:
            
#         """
#         query_qkv_all=[]
#         query_shape_all=[]
#         for ii, query in enumerate(query_frame):
#             B,num_ref_clips,cx,hx,wx=query.shape
#             if ii==0:
#                 query_qkv=self.qkv0(query.permute(0,1,3,4,2))
#             elif ii==1:
#                 query_qkv=self.qkv1(query.permute(0,1,3,4,2))
#             elif ii==2:
#                 query_qkv=self.qkv2(query.permute(0,1,3,4,2))
#             elif ii==3:
#                 query_qkv=self.qkv3(query.permute(0,1,3,4,2))
#             query_qkv_all.append(query_qkv.reshape(B,num_ref_clips,hx*wx,self.num_qkv,cx))
#             query_shape_all.append([hx, wx])

#         supp_qkv_all=[]
#         atten_all=[]
#         for ii, supp in enumerate(supp_frame):
#             B,num_ref_clips,cx,hx,wx=supp.shape
#             if ii==0:
#                 supp_qkv=self.qkv0(supp.permute(0,1,3,4,2))
#             elif ii==1:
#                 supp_qkv=self.qkv1(supp.permute(0,1,3,4,2))
#             elif ii==2:
#                 supp_qkv=self.qkv2(supp.permute(0,1,3,4,2))
#             elif ii==3:
#                 supp_qkv=self.qkv3(supp.permute(0,1,3,4,2))
#             supp_qkv_all.append(supp_qkv.reshape(B,num_ref_clips,hx*wx,self.num_qkv,cx))    

#             if ii>0:
#                 atten=torch.matmul(supp_qkv_all[ii][:,:,:,0,:], query_qkv_all[ii][:,:,:,1,:].transpose(2,3))    ## B*(num_clips-1)*(hx*wx)*(hx*wx)
#                 atten=atten.reshape(B*atten.shape[1],1,hx,wx,query_shape_all[ii][0],query_shape_all[ii][1])    ## (B*(num_clips-1))*1*hx*wx*hx*wx
#                 atten_all.append(atten)
#         #         print(atten.shape) # [60, 60, 30, 30],[30, 30, 30, 30],[15, 15, 15, 15]
#         # print(len(atten_all))
#         atten_new=self.hpn(atten_all)
#         B,num_ref_clips,_,_,_=query_frame[0].shape
#         atten_new=atten_new.reshape(B,num_ref_clips,atten_new.shape[-2],atten_new.shape[-1])
#         return atten_new
        

