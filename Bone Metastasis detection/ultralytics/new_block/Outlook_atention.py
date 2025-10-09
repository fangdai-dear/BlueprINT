import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# https://arxiv.org/pdf/2106.13112
# VOLO Vision Outlooker for Visual Recognition

class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    def __init__(self, dim, num_heads=1, kernel_size=3, padding=1, stride=1,
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        # 计算每个注意力头的维度
        head_dim = dim // num_heads
        self.num_heads = num_heads  # 注意力头的数量
        self.kernel_size = kernel_size  # 卷积核大小
        self.padding = padding  # 卷积填充
        self.stride = stride  # 卷积步幅

        # QK 的缩放因子，默认是头维度的倒数平方根
        self.scale = qk_scale or head_dim ** -0.5

        # 定义线性层，用于计算值 V
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        # 定义线性层，用于计算注意力权重
        self.attn = nn.Linear(dim, kernel_size ** 4 * num_heads)

        # 定义丢弃层，用于注意力计算的丢弃
        self.attn_drop = nn.Dropout(attn_drop)
        # 定义输出投影层
        self.proj = nn.Linear(dim, dim)
        # 定义输出的丢弃层
        self.proj_drop = nn.Dropout(proj_drop)

        # 定义展开操作，将输入特征图转化为局部窗口
        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=padding, stride=stride)
        # 定义平均池化操作，用于生成上下文信息
        self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)

    def forward(self, x):
        # 调整输入维度，从 (B, C, H, W) 转为 (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        B, H, W, C = x.shape  # 解包输入特征图的维度

        # 计算值 V，并调整维度为 (B, C, H, W)
        v = self.v(x).permute(0, 3, 1, 2)

        # 计算经过步幅处理后的特征图高度和宽度
        h, w = math.ceil(H / self.stride), math.ceil(W / self.stride)

        # 将值 V 展开为局部窗口，调整形状为 (B, H, N, kxk, C/H)
        v = self.unfold(v).reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H

        # 对输入特征图进行平均池化，生成上下文信息
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        # 计算注意力权重并调整形状为 (B, H, N, kxk, kxk)
        attn = self.attn(attn).reshape(
            B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
               self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)

        # 缩放注意力权重并进行 softmax 归一化
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # 应用丢弃

        # 使用注意力权重对值 V 进行加权求和
        x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
            B, C * self.kernel_size * self.kernel_size, h * w)

        # 将特征图重构为原始尺寸
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)

        # 通过线性层进行输出投影
        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)  # 应用丢弃

        # 将输出维度调整回 (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x  # 返回处理后的特征图

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class Bottleneck_OutlookAttention(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.cv3 = OutlookAttention( c2, 4)
        self.add = shortcut and c1 == c2

        # self.cv1 = Conv(c1, c_, k[0], 1)
        # self.cv2 = Conv(c_, c2, k[1], 1, g=g)


    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck_OutlookAttention(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

# 在c3k=True时，使用Bottleneck_StarsBlock特征融合，为false的时候我们使用普通的Bottleneck提取特征
class C3k2_OutlookAttention(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


# class PSABlock_OutlookAttention(nn.Module):
#     """
#     PSABlock class implementing a Position-Sensitive Attention block for neural networks.
#
#     This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
#     with optional shortcut connections.
#
#     Attributes:
#         attn (Attention): Multi-head attention module.
#         ffn (nn.Sequential): Feed-forward neural network module.
#         add (bool): Flag indicating whether to add shortcut connections.
#
#     Methods:
#         forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.
#
#     Examples:
#         Create a PSABlock and perform a forward pass
#         >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
#         >>> input_tensor = torch.randn(1, 128, 32, 32)
#         >>> output_tensor = psablock(input_tensor)
#     """
#
#     def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
#         """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
#         super().__init__()
#
#         self.attn = OutlookAttention( c, 4)
#         self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
#         self.add = shortcut
#
#     def forward(self, x):
#         """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
#         x = x + self.attn(x) if self.add else self.attn(x)
#         x = x + self.ffn(x) if self.add else self.ffn(x)
#         return x
#
#
#
# class C2PSA_OutlookAttention(nn.Module):
#     """
#     C2PSA module with attention mechanism for enhanced feature extraction and processing.
#
#     This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
#     capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
#
#     Attributes:
#         c (int): Number of hidden channels.
#         cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
#         cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
#         m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.
#
#     Methods:
#         forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.
#
#     Notes:
#         This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.
#
#     Examples:
#         >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
#         >>> input_tensor = torch.randn(1, 256, 64, 64)
#         >>> output_tensor = c2psa(input_tensor)
#     """
#
#     def __init__(self, c1, c2, n=1, e=0.5):
#         """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
#         super().__init__()
#         assert c1 == c2
#         self.c = int(c1 * e)
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c1, 1)
#
#         self.m = nn.Sequential(*(PSABlock_OutlookAttention(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
#
#     def forward(self, x):
#         """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
#         a, b = self.cv1(x).split((self.c, self.c), dim=1)
#         b = self.m(b)
#         return self.cv2(torch.cat((a, b), 1))




if __name__ =='__main__':
    stars_Block =OutlookAttention(256)
    #创建一个输入张量，形状为(batch_size, H*W,C)
    batch_size = 8
    input_tensor=torch.randn(batch_size, 256, 64, 64 )
    #运行模型并打印输入和输出的形状
    output_tensor =stars_Block(input_tensor)
    print("Input shape:",input_tensor.shape)
    print("0utput shape:",output_tensor.shape)


