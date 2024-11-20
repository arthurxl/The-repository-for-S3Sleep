import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from copy import deepcopy
from tools import zigzag_path
from autocorr import AutoCorrelation
from attention import FlowAttention, FlashAttention, ProbAttention
from einops import repeat

########################################################################################
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class GELU(nn.Module):
    # for older versions of PyTorch.  For new versions you can use nn.GELU() instead.
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        x = torch.nn.functional.gelu(x)
        return x


class DUCNN(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(DUCNN, self).__init__()
        drate = 0.5
        # drate = 0
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=4, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        return x_concat


##########################################################################################


def attention(query, key, value, dropout=None):
    "Implementation of Scaled dot product attention"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 groups=1,
                 bias=True):
        self.__padding = (kernel_size - 1) * dilation

        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias)

    def forward(self, input):
        result = super(CausalConv1d, self).forward(input)

        if self.__padding != 0:
            return result[:, :, :-self.__padding]
        return result


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h

        self.convs = clones(CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1), 3)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        # self.attention = AutoCorrelation(attention_dropout=dropout)
        # self.attention = FlowAttention(dropout)
        self.attention = ProbAttention(attention_dropout=dropout)
        # self.attention = FlashAttention(attention_dropout=dropout)
        self.freqs_cis = precompute_freqs_cis(d_model, afr_reduced_cnn_size).to(device)

    def forward(self, query, key, value):
        "Implements Multi-head attention"
        nbatches = query.size(0)
        query, key = apply_rotary_emb(query, key, freqs_cis=self.freqs_cis)
        query = query.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.convs[1](key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.convs[2](value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        # x, self.attn = attention(query, key, value, dropout=self.dropout)
        x, self.attn = self.attention(query, key, value, None)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        return self.linear(x)


class LayerNorm(nn.Module):
    "Construct a layer normalization module."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerOutput(nn.Module):
    '''
    A residual connection followed by a layer norm.
    '''

    def __init__(self, size, dropout):
        super(SublayerOutput, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
        # return self.norm(x + self.dropout(sublayer(x)))


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TCE(nn.Module):
    '''
    Transformer Encoder

    It is a stack of N layers.
    '''

    def __init__(self, layer, N):
        super(TCE, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer

    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''

    def __init__(self, size, self_attn, feed_forward, afr_reduced_cnn_size, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer_output = clones(SublayerOutput(size, dropout), 2)
        self.size = size
        self.conv = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x_in):
        "Transformer Encoder"

        query = self.conv(x_in)
        x = self.sublayer_output[0](query, lambda x: self.self_attn(query, x_in, x_in))  # Encoder self-attention
        return self.sublayer_output[1](x, self.feed_forward)


class PositionwiseFeedForward(nn.Module):
    "Positionwise feed-forward network."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        "Implements FFN equation."
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class S3Sleep(nn.Module):
    def __init__(self):
        super(S3Sleep, self).__init__()

        # N = 2  # number of TCE clones
        N = 3
        # d_model = 80  # set to be 100 for SHHS dataset
        d_model = 80
        d_ff = 120  # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30

        self.mrcnn = DUCNN(afr_reduced_cnn_size)  # use MRCNN_SHHS for SHHS dataset
        # self.mrcnn = MRCNN_SHHS(afr_reduced_cnn_size)
        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size + 1)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size + 1, dropout), N)
        self.cls = nn.Parameter(torch.randn(1, 1, 80))
        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)
        # self.cattn = CrossAttention(h, d_model, afr_reduced_cnn_size, d_ff)
        # self.ada = AdaFormer()
        # todo 自行添加之字形扫描
        self.zig_path = zigzag_path(5, 6)
        print("[INFO] Number Of Attn Sleep Parameters:{}".format(
            sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x_feat = self.mrcnn(x)
        # x_1, _ = self.ada(x_feat)
        x_feat = x_feat[:, self.zig_path[5], :]
        # x_1 = x_1.squeeze(2)
        cls = repeat(self.cls, "() n d -> b n d", b=x_feat.shape[0])
        x_feat = torch.cat((x_feat, cls), dim=1)
        # x_feat_1 = x_feat[:, self.zig_path[5], :]
        # x_feat_2 = x_feat[:, self.zig_path[4], :]
        # muti scale 和短时傅里叶变换添加
        # encoded_features = x_feat
        # x_feat = self.revin(x_feat.transpose(1, 2), mode='norm').transpose(1, 2)
        # dispatcher
        encoded_features = self.tce(x_feat)[:, :30, :]
        # encoded_features = self.cattn(x_1, encoded_features)
        # encoded_features_1 = self.tce(x_feat_1)
        # encoded_features_2 = self.tce(x_feat_2)
        # encoded_features = (encoded_features + encoded_features_1 + encoded_features_2) / 3
        # encoded_features = self.revin(encoded_features.transpose(1, 2), mode='denorm').transpose(1, 2)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        final_output = self.fc(encoded_features)
        return final_output


class CrossAttention(nn.Module):
    def __init__(self, h, d_model, afr_reduced_cnn_size, dff, dropout=0.1):
        super(CrossAttention, self).__init__()
        self.muti = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, dff),
                                 nn.Dropout(dropout),
                                 nn.ReLU(inplace=True),
                                 nn.Linear(dff, d_model))
        self.wq = CausalConv1d(afr_reduced_cnn_size, afr_reduced_cnn_size, kernel_size=7, stride=1, dilation=1)

    def forward(self, x1, x2):
        x1 = self.wq(x1)
        x = self.muti(x1, x2, x2)
        out = x2 + self.norm1(x)
        x = self.ffn(out)

        return out + self.norm2(x)


def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 计算词向量元素两两分组之后，每组元素对应的旋转角度
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成 token 序列索引 t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()
    # torch.polar 的文档
    # https://pytorch.org/docs/stable/generated/torch.polar.html
    # 计算结果是个复数向量
    # 假设 freqs = [x, y]
    # 则 freqs_cis = [cos(x) + sin(x)i, cos(y) + sin(y)i]
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
):
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim // 2, 2]
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # 转为复数域
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    # 应用旋转操作，然后将结果转回实数域
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class S3Sleep_pa(nn.Module):
    def __init__(self):
        super(S3Sleep_pa, self).__init__()

        # N = 2  # number of TCE clones
        N = 2
        # d_model = 80  # set to be 100 for SHHS dataset
        d_model = 100
        d_ff = 120  # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30
        self.mrcnn = DUCNN_SHHS(afr_reduced_cnn_size)  # use MRCNN_SHHS for SHHS dataset

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        # fft_attn = MultiHeadedAttention(5, 80, 30)
        # fft_ff = PositionwiseFeedForward(80, 120, dropout)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), N)
        # self.fft_tce = TCE(EncoderLayer(80, deepcopy(fft_attn), deepcopy(fft_ff), 30, dropout), N)
        # self.fft_tce = nn.Identity()
        # self.embed = PatchEmbed((30, 80), (10, 10))
        # self.ada = AdaFormer()
        # self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)
        self.fc = nn.Sequential(nn.Linear(d_model * afr_reduced_cnn_size, 512),
                                nn.ReLU(),
                                nn.Linear(512, num_classes))
        attn_intra = MultiHeadedAttention(10, 3000, 30)
        ff_intra = PositionwiseFeedForward(3000, 4800, dropout)
        self.tce_intra = TCE(EncoderLayer(3000, deepcopy(attn_intra), deepcopy(ff_intra), 30, dropout), 2)
        # todo 自行添加之字形扫描
        self.zig_path = zigzag_path(5, 6)
        self.gru = nn.GRU(3000, 3000, batch_first=True)
        print("[INFO] Number Of Attn Sleep Parameters:{}".format(
            sum([p.numel() for p in self.parameters() if p.requires_grad])))

    def forward(self, x):
        x = x.split(1, 1)
        output_list = []
        # ban_loss = 0
        for cur_x in x:
            x_feat = self.mrcnn(cur_x)
            # encoded_features = x_feat
            x_feat = x_feat[:, self.zig_path[4], :]
            # x_feat = self.revin(x_feat.transpose(1, 2), mode='norm').transpose(1, 2)
            encoded_features = self.tce(x_feat)
            # encoded_features_1, cur_los = self.ada(x_feat)
            # ban_loss += cur_los
            # x_fft = torch.fft.fft(x_feat).real.contiguous()
            # print(x_fft.shape)
            # x_fft = self.embed(x_fft).contiguous()
            # fft_feature = self.fft_tce(x_fft).view(encoded_features.shape[0], 1, -1)
            # encoded_features = self.revin(encoded_features.transpose(1, 2), mode='denorm').transpose(1, 2)
            encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], 1, -1)
            # encoded_features_1 = encoded_features_1.contiguous().view(encoded_features.shape[0], 1, -1)
            output_list.append(encoded_features)
        out = torch.cat(output_list, dim=1)
        out = self.tce_intra(out)
        # out = self.gru(out)[0]
        final_output = self.fc(out)
        return final_output


class PatchEmbed(nn.Module):
    def __init__(self, input_size, patch_size, norm_layer=None):
        super(PatchEmbed, self).__init__()
        self.grid_size = (input_size[0] / patch_size[0], input_size[1] / patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.norm = norm_layer(patch_size[0] * patch_size[1]) if norm_layer else nn.Identity()
        self.proj = nn.Conv2d(1, patch_size[0] * patch_size[1], kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x.unsqueeze(1)).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


######################################################################

class DUCNN_SHHS(nn.Module):
    def __init__(self, afr_reduced_cnn_size):
        super(DUCNN_SHHS, self).__init__()
        drate = 0.5
        self.GELU = GELU()  # for older versions of PyTorch.  For new versions use nn.GELU() instead.
        self.features1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=8, stride=2, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=4, stride=5, padding=2)
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=400, stride=50, bias=False, padding=200),
            nn.BatchNorm1d(64),
            self.GELU,
            nn.MaxPool1d(kernel_size=4, stride=2, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.Conv1d(128, 128, kernel_size=6, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.GELU,

            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )
        self.dropout = nn.Dropout(drate)
        self.inplanes = 128
        self.AFR = self._make_layer(SEBasicBlock, afr_reduced_cnn_size, 1)

    def _make_layer(self, block, planes, blocks, stride=1):  # makes residual SE block
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.features1(x)
        x2 = self.features2(x)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        return x_concat





class S3(nn.Module):
    def __init__(self):
        super(S3, self).__init__()
        N = 2  # number of TCE clones
        d_model = 170  # set to be 100 for SHHS dataset
        d_ff = 200  # dimension of feed forward
        h = 5  # number of attention heads
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30
        seq_len = 21

        self.mrcnn = DUCNN_SHHS(afr_reduced_cnn_size)  # use MRCNN_SHHS for SHHS dataset

        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size + 1)
        intra_attn = MultiHeadedAttention(h, d_model, seq_len)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size + 1, dropout), N)
        self.cls = nn.Parameter(torch.randn(1, 1, d_model), requires_grad=True).to(device)
        # transformer_layer = TransformerEncoderLayer(d_model=d_model, nhead=5, dim_feedforward=d_ff, dropout=dropout,
        #                                             batch_first=True)
        # self.intra_transformer = TransformerEncoder(transformer_layer, N)
        self.intra_transformer = TCE(EncoderLayer(d_model, deepcopy(intra_attn), deepcopy(ff), seq_len, dropout),
                                     N)
        self.seq_out = nn.Sequential(nn.Linear(d_model, 50),
                                     nn.ReLU(),
                                     nn.Linear(50, num_classes))
        self.zigpath = zigzag_path(5, 6)

    def forward(self, x):
        b = x.size(0)
        l = x.size(1)
        encode_features = []
        for seq_idx in range(l):
            epoch_x = x[:, seq_idx:seq_idx + 1, :]

            epoch_x = self.mrcnn(epoch_x)
            epoch_x = epoch_x[:, self.zigpath[4], :]
            # encode_feature = self.tce(epoch_x)
            cls = repeat(self.cls, "() l d -> b l d", b=b)
            encode_feature = torch.cat([epoch_x, cls], dim=1)
            encode_feature = self.tce(encode_feature)
            encode_features.append(encode_feature[:, -1, :].unsqueeze(1))
        encode_features = torch.cat(encode_features, dim=1)
        out = self.intra_transformer(encode_features)
        out = self.seq_out(out)
        return out


