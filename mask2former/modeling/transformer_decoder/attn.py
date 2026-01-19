import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init

def conv_layer(in_dim, out_dim, kernel_size=1, padding=0, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_dim), nn.ReLU(True))


def linear_layer(in_dim, out_dim, bias=False):
    return nn.Sequential(nn.Linear(in_dim, out_dim, bias),
                         nn.BatchNorm1d(out_dim), nn.ReLU(True))


class CoordConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.conv1 = conv_layer(in_channels + 2, out_channels, kernel_size,
                                padding, stride)

    def add_coord(self, input):
        b, _, h, w = input.size()
        x_range = torch.linspace(-1, 1, w, device=input.device)
        y_range = torch.linspace(-1, 1, h, device=input.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([b, 1, -1, -1])
        x = x.expand([b, 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        input = torch.cat([input, coord_feat], 1)
        return input

    def forward(self, x):
        x = self.add_coord(x)
        x = self.conv1(x)
        return x


class Projector(nn.Module):
    def __init__(self, word_dim=1024, in_dim=256, kernel_size=3):
        super().__init__()
        self.in_dim = in_dim
        self.kernel_size = kernel_size
        # visual projector
        self.vis = nn.Sequential(  # os16 -> os4
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim * 2, 3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            conv_layer(in_dim * 2, in_dim, 3, padding=1),
            nn.Conv2d(in_dim, in_dim, 1))
        # textual projector
        out_dim = 1 * in_dim * kernel_size * kernel_size + 1
        self.txt = nn.Linear(word_dim, out_dim)

    def forward(self, x, word):
        '''
            x: b, 512, 26, 26
            word: b, 512
        '''
        x = self.vis(x)
        B, C, H, W = x.size()
        # 1, b*256, 104, 104
        x = x.reshape(1, B * C, H, W)
        # txt: b, (256*3*3 + 1) -> b, 256, 3, 3 / b
        word = self.txt(word)
        weight, bias = word[:, :-1], word[:, -1]
        weight = weight.reshape(B, C, self.kernel_size, self.kernel_size)
        # Conv2d - 1, b*256, 104, 104 -> 1, b, 104, 104
        out = F.conv2d(x,
                       weight,
                       padding=self.kernel_size // 2,
                       groups=weight.size(0),
                       bias=bias)
        out = out.transpose(0, 1)
        # b, 1, 104, 104
        return out


class TransformerDecoder(nn.Module):
    def __init__(self,
                 num_layers,
                 d_model,
                 nhead,
                 dim_ffn,
                 dropout,
                 return_intermediate=False):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model=d_model,
                                    nhead=nhead,
                                    dim_feedforward=dim_ffn,
                                    dropout=dropout) for _ in range(num_layers)
        ])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def forward(self, vis, txt, pad_mask):
        '''
            vis: b, 512, h, w
            txt: b, L, 512
            pad_mask: b, L
        '''
        B, C, H, W = vis.size()
        _, L, D = txt.size()
        # position encoding
        vis_pos = self.pos2d(C, H, W)
        txt_pos = self.pos1d(D, L)
        # reshape & permute
        vis = vis.reshape(B, C, -1).permute(2, 0, 1)
        txt = txt.permute(1, 0, 2)
        # forward
        output = vis
        intermediate = []
        for layer in self.layers:
            output = layer(output, txt, vis_pos, txt_pos, pad_mask)
            if self.return_intermediate:
                # HW, b, 512 -> b, 512, HW
                intermediate.append(self.norm(output).permute(1, 2, 0))

        if self.norm is not None:
            # HW, b, 512 -> b, 512, HW
            output = self.norm(output).permute(1, 2, 0)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
                # [output1, output2, ..., output_n]
                return intermediate
            else:
                # b, 512, HW
                return output
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model=512,
                 nhead=9,
                 dim_feedforward=2048,
                 dropout=0.1):
        super().__init__()
        # Normalization Layer
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        # Attention Layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model,
                                                    nhead,
                                                    dropout=dropout,
                                                    kdim=d_model,
                                                    vdim=d_model)
        # FFN
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(True), nn.Dropout(dropout),
                                 nn.LayerNorm(dim_feedforward),
                                 nn.Linear(dim_feedforward, d_model))
        # LayerNorm & Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, vis, txt, vis_pos, txt_pos, pad_mask):
        '''
            vis: 26*26, b, 512
            txt: L, b, 512
            vis_pos: 26*26, 1, 512
            txt_pos: L, 1, 512
            pad_mask: b, L
        '''
        # Self-Attention
        vis2 = self.norm1(vis)
        q = k = self.with_pos_embed(vis2, vis_pos)
        vis2 = self.self_attn(q, k, value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis = vis + self.dropout1(vis2)
        # Cross-Attention
        vis2 = self.norm2(vis)
        vis2 = self.multihead_attn(query=self.with_pos_embed(vis2, vis_pos),
                                   key=self.with_pos_embed(txt, txt_pos),
                                   value=txt,
                                   key_padding_mask=pad_mask)[0]
        vis2 = self.cross_attn_norm(vis2)
        vis = vis + self.dropout2(vis2)
        # FFN
        vis2 = self.norm3(vis)
        vis2 = self.ffn(vis2)
        vis = vis + self.dropout3(vis2)
        return vis


class FPN(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 1024],
                 out_channels=[256, 512, 1024]):
        super(FPN, self).__init__()
        # text projection
        self.txt_proj = linear_layer(in_channels[2], out_channels[2])
        # fusion 1: v5 & seq -> f_5: b, 1024, 13, 13
        self.f1_v_proj = conv_layer(in_channels[2], out_channels[2], 1, 0)
        self.norm_layer = nn.Sequential(nn.BatchNorm2d(out_channels[2]),
                                        nn.ReLU(True))
        # fusion 2: v4 & fm -> f_4: b, 512, 26, 26
        self.f2_v_proj = conv_layer(in_channels[1], out_channels[1], 3, 1)
        self.f2_cat = conv_layer(out_channels[2] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 3: v3 & fm_mid -> f_3: b, 512, 52, 52
        self.f3_v_proj = conv_layer(in_channels[0], out_channels[0], 3, 1)
        self.f3_cat = conv_layer(out_channels[0] + out_channels[1],
                                 out_channels[1], 1, 0)
        # fusion 4: f_3 & f_4 & f_5 -> fq: b, 256, 26, 26
        self.f4_proj5 = conv_layer(out_channels[2], out_channels[1], 3, 1)
        self.f4_proj4 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        self.f4_proj3 = conv_layer(out_channels[1], out_channels[1], 3, 1)
        # aggregation
        self.aggr = conv_layer(3 * out_channels[1], out_channels[1], 1, 0)
        self.coordconv = nn.Sequential(
            CoordConv(out_channels[1], out_channels[1], 3, 1),
            conv_layer(out_channels[1], out_channels[1], 3, 1))

    def forward(self, imgs, state):
        # v3, v4, v5: 256, 52, 52 / 512, 26, 26 / 1024, 13, 13
        v3, v4, v5 = imgs
        # fusion 1: b, 1024, 13, 13
        # text projection: b, 1024 -> b, 1024
        state = self.txt_proj(state).unsqueeze(-1).unsqueeze(
            -1)  # b, 1024, 1, 1
        f5 = self.f1_v_proj(v5)
        f5 = self.norm_layer(f5 * state)
        # fusion 2: b, 512, 26, 26
        f4 = self.f2_v_proj(v4)
        f5_ = F.interpolate(f5, scale_factor=2, mode='bilinear')
        f4 = self.f2_cat(torch.cat([f4, f5_], dim=1))
        # fusion 3: b, 256, 26, 26
        f3 = self.f3_v_proj(v3)
        f3 = F.avg_pool2d(f3, 2, 2)
        f3 = self.f3_cat(torch.cat([f3, f4], dim=1))
        # fusion 4: b, 512, 13, 13 / b, 512, 26, 26 / b, 512, 26, 26
        fq5 = self.f4_proj5(f5)
        fq4 = self.f4_proj4(f4)
        fq3 = self.f4_proj3(f3)
        # query
        fq5 = F.interpolate(fq5, scale_factor=2, mode='bilinear')
        fq = torch.cat([fq3, fq4, fq5], dim=1)
        fq = self.aggr(fq)
        fq = self.coordconv(fq)
        # b, 512, 26, 26
        return fq


class PWAM(nn.Module):
    def __init__(self, dim, v_in_channels, l_in_channels, key_channels, value_channels, num_heads=1, dropout=0.0):
        super(PWAM, self).__init__()
        # input x shape: (B, H*W, dim)
        self.vis_project = nn.Sequential(nn.Conv1d(dim, dim, 1, 1),  # the init function sets bias to 0 if bias is True
                                         nn.GELU(),
                                         nn.Dropout(dropout)
                                        )

        key_channels = v_in_channels
        value_channels = v_in_channels
        self.image_lang_att = SpatialImageLanguageAttention(v_in_channels,  # v_in
                                                            l_in_channels,  # l_in
                                                            key_channels,  # key
                                                            value_channels,  # value
                                                            out_channels=value_channels,  # out
                                                            num_heads=num_heads)

        self.project_mm = nn.Sequential(nn.Conv1d(value_channels, value_channels, 1, 1),
                                        nn.GELU(),
                                        nn.Dropout(dropout)
                                        )

    def forward(self, x, l, l_mask):
        # input x shape: (B, dim, H*W)
        vis = self.vis_project(x)  # (B, dim, H*W)

        lang = self.image_lang_att(x, l, l_mask)  # (B, H*W, dim)

        lang = lang.permute(0, 2, 1)  # (B, dim, H*W)

        mm = torch.mul(vis, lang)
        mm = self.project_mm(mm)  # (B, dim, H*W)

        mm = mm.permute(0, 2, 1)  # (B, H*W, dim)

        return mm


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

        ### initialize these layers
        weight_init.c2_xavier_fill(self.f_key[0])
        weight_init.c2_xavier_fill(self.f_query[0])
        weight_init.c2_xavier_fill(self.f_value[0])
        weight_init.c2_xavier_fill(self.W[0])

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)

        # import pdb; pdb.set_trace()
        B, HW = x.size(0), x.size(2)
        # x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        # l = l.permute(0, 2, 1)  

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out


from typing import Optional
import torch
import torch.nn as nn
# from .flash_attn import UniMultiHeadAttention

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class TransFusion(torch.nn.Module):
    def __init__(self, d_model=256, n_heads=8, dropout=0.1, fusion_method='attn adain gate'):
        super(TransFusion, self).__init__()
        
        self.norm_v = nn.LayerNorm(d_model)
        self.lang_proj = nn.Linear(768, d_model)

        self.norm_l = nn.LayerNorm(d_model)
        fusion_method = 'attn adain gate'
        # if 'attn' in fusion_method:
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model,
                                        n_heads,
                                        dropout=dropout,
                                        kdim=d_model,
                                        vdim=d_model)
        
        self.fusion_method = fusion_method
        # if 'adain' in fusion_method:
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 3 * d_model, bias=True)
        )
        # zero init
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

        # if 'gate' in fusion_method:
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False), nn.ReLU(),
            nn.Linear(d_model, d_model, bias=False), nn.Tanh())

    @staticmethod
    def pos1d(d_model, length):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                              -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe.unsqueeze(1)  # n, 1, 512

    @staticmethod
    def pos2d(d_model, height, width):
        """
        :param d_model: dimension of the model
        :param height: height of the positions
        :param width: width of the positions
        :return: d_model*height*width position matrix
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(
            torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
            0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
            0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos.to(tensor.device)

    def forward(self, srcs, lang_dict_features, spatial_shapes):
        new_srcs = []

        ref, mask = lang_dict_features["word_embeds"], lang_dict_features["masks"]
        mask = ~mask.bool()   # reverse mask to valid locations are 1
        # if 'attn' in self.fusion_method:
        ref = self.lang_proj(ref)
        ref = self.norm_l(ref)
        lang_c = lang_dict_features["sent_embeds"] 
        lang_c = self.lang_proj(lang_c)

        _, L, D = ref.size()
        txt_pos = self.pos1d(D, L)
        ref = ref.transpose(0,1)

        # for each level
        # src: q, ref: k, ref_values: v
        for i, src in enumerate(srcs):
            src = self.norm_v(src)
            src = src.transpose(0, 1)

            # if 'attn' in self.fusion_method:
            src_l = self.cross_attn(query=self.with_pos_embed(src.transpose(0,1), None), 
                                    key=self.with_pos_embed(ref, txt_pos), 
                                    value=ref, 
                                    key_padding_mask=mask)[0]
            src_l_1 = self.cross_attn_norm(src_l)
            src_l = src_l_1 + src_l
            src_l = src_l.transpose(0,1)
            # else:
            #     src_l = src 
            
            # if 'adain' in self.fusion_method:
            shift_msa_l, scale_msa_l, gate_msa_l = self.adaLN_modulation(lang_c).chunk(3, dim=1)
            src_l = gate_msa_l.unsqueeze(1) * modulate(src_l, shift_msa_l, scale_msa_l)
                
            # if 'gate' in self.fusion_method:
            src = src + (self.gate(src_l) * src_l)
            # else:
            #     src = src + src_l 

            new_srcs.append(src.transpose(0, 1))

        return new_srcs



# class TransFusion_re(torch.nn.Module):
#     def __init__(self, d_model=256, n_heads=8, dropout=0.1, fusion_method=None):
#         super(TransFusion_re, self).__init__()
        
#         self.norm_v = nn.LayerNorm(d_model)
#         self.lang_proj = nn.Linear(768, d_model)

#         if 'attn' in fusion_method:
#             self.norm_l = nn.LayerNorm(d_model)
#             self.cross_attn_norm = nn.LayerNorm(d_model)
#             self.cross_attn = nn.MultiheadAttention(d_model,
#                                             n_heads,
#                                             dropout=dropout,
#                                             kdim=d_model,
#                                             vdim=d_model)
#         self.fusion_method = fusion_method
#         if 'adain' in fusion_method:
#             self.adaLN_modulation = nn.Sequential(
#                 nn.SiLU(),
#                 nn.Linear(d_model, 3 * d_model, bias=True)
#             )
#             # zero init
#             nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
#             nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

#         if 'gate' in fusion_method:
#             self.gate1 = nn.Sequential(
#                 nn.Linear(d_model, d_model, bias=False), nn.ReLU(),
#                 nn.Linear(d_model, d_model, bias=False), nn.Tanh())

#     @staticmethod
#     def pos1d(d_model, length):
#         """
#         :param d_model: dimension of the model
#         :param length: length of positions
#         :return: length*d_model position matrix
#         """
#         if d_model % 2 != 0:
#             raise ValueError("Cannot use sin/cos positional encoding with "
#                              "odd dim (got dim={:d})".format(d_model))
#         pe = torch.zeros(length, d_model)
#         position = torch.arange(0, length).unsqueeze(1)
#         div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
#                               -(math.log(10000.0) / d_model)))
#         pe[:, 0::2] = torch.sin(position.float() * div_term)
#         pe[:, 1::2] = torch.cos(position.float() * div_term)

#         return pe.unsqueeze(1)  # n, 1, 512

#     @staticmethod
#     def pos2d(d_model, height, width):
#         """
#         :param d_model: dimension of the model
#         :param height: height of the positions
#         :param width: width of the positions
#         :return: d_model*height*width position matrix
#         """
#         if d_model % 4 != 0:
#             raise ValueError("Cannot use sin/cos positional encoding with "
#                              "odd dimension (got dim={:d})".format(d_model))
#         pe = torch.zeros(d_model, height, width)
#         # Each dimension use half of d_model
#         d_model = int(d_model / 2)
#         div_term = torch.exp(
#             torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
#         pos_w = torch.arange(0., width).unsqueeze(1)
#         pos_h = torch.arange(0., height).unsqueeze(1)
#         pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
#             0, 1).unsqueeze(1).repeat(1, height, 1)
#         pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
#             0, 1).unsqueeze(1).repeat(1, height, 1)
#         pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
#             0, 1).unsqueeze(2).repeat(1, 1, width)
#         pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
#             0, 1).unsqueeze(2).repeat(1, 1, width)

#         return pe.reshape(-1, 1, height * width).permute(2, 1, 0)  # hw, 1, 512

#     def with_pos_embed(self, tensor, pos):
#         return tensor if pos is None else tensor + pos.to(tensor.device)

#     def forward(self, srcs, lang_dict_features):
#         new_srcs = []

#         ref, mask = lang_dict_features["word_embeds"], lang_dict_features["masks"]
#         mask = ~mask.bool()   # reverse mask to valid locations are 1
#         if 'attn' in self.fusion_method:
#             ref = self.lang_proj(ref)
#             ref = self.norm_l(ref)
#         lang_c = lang_dict_features["sent_embeds"] 
#         lang_c = self.lang_proj(lang_c)

#         _, L, D = ref.size()
#         txt_pos = self.pos1d(D, L)
#         ref = ref.transpose(0,1)

#         # for each level
#         # src: q, ref: k, ref_values: v
#         for i, src in enumerate(srcs):
#             _, c, hi, wi = src.shape
#             src = src.flatten(-2).permute(0, 2, 1)  # [b, hiwi, c]
#             # hi, wi = spatial_shapes[i]
#             c = src.shape[-1]
#             # vis_pos = self.pos2d(c, hi, wi)

#             src = self.norm_v(src)

#             shift_msa_l, scale_msa_l, gate_msa_l = self.adaLN_modulation(lang_c).chunk(3, dim=1)
#             lang_c_filtered = gate_msa_l.unsqueeze(1) * modulate(ref.transpose(0,1), shift_msa_l, scale_msa_l)
                
#             lang_c_filtered = lang_c_filtered + (self.gate1(lang_c_filtered) * lang_c_filtered)
#             lang_c_filtered = lang_c_filtered.transpose(0, 1)
        
#             src_l = self.cross_attn(query=self.with_pos_embed(src.transpose(0,1), None),  key=self.with_pos_embed(lang_c_filtered, txt_pos),  value=lang_c_filtered, key_padding_mask=mask)[0]
#             src_l_1 = self.cross_attn_norm(src_l)
#             src_l = src_l_1 + src_l
#             src_l = src_l.transpose(0,1)

#             new_srcs.append(src)
#         return new_srcs
