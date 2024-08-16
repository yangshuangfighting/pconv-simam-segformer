import torch
import torch.nn as nn
import math

class MS3DA(nn.Module):
    """ Our proposed Global Channel Interaction Attention Module with structural re-parameterization.
        You can boost the inference speed by switching the mode to inference with "switch_to_deploy" function.
        Args:
            channel: the number of channels
            deploy: training mode or inference mode
        Return:
            return a refined feature map
    """
    def __init__(self, channel, deploy = False):
        super(MS3DA, self).__init__()
        #Global Average Pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #X-Avg Pool && Y-Avg Pool
        self.pool_h=nn.AdaptiveAvgPool2d((None,1))
        self.pool_w=nn.AdaptiveAvgPool2d((1,None))
        #Sigmoid Activation Function
        self.act = nn.Sigmoid()
        #Channel attention branch
        self.c_branch = c_branch(channel, deploy = deploy)
        #Spatial attention branch
        self.sp_branch = sp_branch(channel, deploy = deploy)
        #Batch Norm
        self.bn = nn.BatchNorm2d(channel)
    def forward(self, x):
        _, _, H, W = x.shape
        #Shared Coordinate Embedding Submodule
        #N*C*H*W -> N*C*H*1; N*C*H*W -> N*C*W*1
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        #[N*C*H*1; N*C*W*1] -> N*C*(H+W)*1
        out = torch.cat([x_h, x_w], dim=2)
        #N*C*(H+W)*1 -> N*1*(H+W)
        emb = torch.mean(out, dim=1, keepdim=True).squeeze(-1)
        #Channel attention branch
        emb = self.sp_branch(emb)
        #N*1*(H+W) -> N*H*1 && N*1*W
        h_emb, w_emb = torch.split(emb, [H, W], dim=2)
        h_emb = h_emb.permute(0, 2, 1)

        #Spatial attention branch
        sp = (h_emb*w_emb).unsqueeze(-3)
        #N*C*H*W -> N*C*1*1 -> N*1*C
        c_credit = self.avg_pool(x_h).squeeze(-1).permute(0, 2, 1)

        #Channel attention branch
        #N*1*C -> N*C*1
        c_credit = self.c_branch(c_credit).permute(0, 2, 1)
        #N*C*1 -> N*C*1*1
        c_credit = (c_credit).unsqueeze(-1)
        #MS3DA Attention Generation Submodule 
        attention = self.act(self.bn(sp * c_credit))
        #Refine the input feature map
        return x * attention

#Spatial Attention Branch
class sp_branch(nn.Module):
    """ Spatial Attention Branch with structural re-parameterization.
        Args:
            channel: the number of channels
            deploy: training mode or inference mode 
        Return:
            return a directional vector after multi-scale 1-D convolutions 
    """
    def __init__(self, channel, deploy=False):
        super(sp_branch, self).__init__()
        self.deploy = deploy
        self.channel = channel
        if deploy:
            self.sp_reparam = nn.Conv1d(1, 1, kernel_size=5, padding=int(5/2), bias=False) 
        else:
            self.convsp1 = nn.Conv1d(1, 1, kernel_size=3, padding=int(3/2), bias=False)
            self.convsp2 = nn.Conv1d(1, 1, kernel_size=5, padding=int(5/2), bias=False)

    def forward(self, x):
        if hasattr(self, 'sp_reparam'):
            return self.sp_reparam(x)

        return self.convsp1(x) + self.convsp2(x)
    def _pad_3_to_5_kernel(self,kernel3):

        if kernel3 is None:
            return 0
        
        else:
            weight = kernel3.weight.data
            return torch.nn.functional.pad(weight, [1,1,0,0])
        

    def get_equivalent_kernel_bias(self):
        kernal3 = self._pad_3_to_5_kernel(self.convsp1)
        return kernal3 + self.convsp2.weight.data #3->5 + 5

    def switch_to_deploy(self):
        if hasattr(self, 'sp_reparam'):
            return
        kernal = self.get_equivalent_kernel_bias() #3->5
        self.sp_reparam = nn.Conv1d(1, 1, kernel_size=self.convsp2.kernel_size, padding=self.convsp2.padding, bias=False) 
        self.sp_reparam.weight.data = kernal
        self.__delattr__('convsp1')
        self.__delattr__('convsp2')
        self.deploy = True

class c_branch(nn.Module):
    """ Channel Attention Branch with structural re-parameterization.
        Args:
            channel: the number of channels
            deploy: training mode or inference mode 
        Return:
            return a directional vector after multi-scale 1-D convolutions 
    """
    def __init__(self, channel, deploy=False):
        super(c_branch, self).__init__()
        self.deploy = deploy
        self.channel = channel
        t = int(abs((math.log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1

        self.k_max = max(5,k)
        if deploy:
            self.c_reparam = nn.Conv1d(1, 1, kernel_size=self.k_max, padding=int(self.k_max/2), bias=False) 
        else:
            self.convc1 = nn.Conv1d(1, 1, kernel_size=3, padding=int(3/2), bias=False)
            self.convc2 = nn.Conv1d(1, 1, kernel_size=5, padding=int(5/2), bias=False)
            self.convc3 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
    def forward(self, x):
        if hasattr(self, 'c_reparam'):
            return self.c_reparam(x)

        return self.convc1(x) + self.convc2(x) + self.convc3(x) 
    #Pad the kernel size to k_max
    def _pad_to_max_kernel(self, kernel):
        weight = kernel.weight.data
        k_size = kernel.kernel_size
        pad_time = (self.k_max - k_size[0])//2
        if kernel is None:
            return 0
        else:
            return torch.nn.functional.pad(weight, [pad_time, pad_time, 0, 0])

    def get_equivalent_kernel_bias(self):
        kernal3 = self._pad_to_max_kernel(self.convc1)
        kernal5 = self._pad_to_max_kernel(self.convc2)
        kernalk = self._pad_to_max_kernel(self.convc3)
        return kernal3 + kernal5 + kernalk#3->k_max + 5->k_max + k->max

    def switch_to_deploy(self):
        if hasattr(self, 'c_reparam'):
            return
        kernal = self.get_equivalent_kernel_bias() #3->k_max,5->k_max,k->k_max

        self.c_reparam = nn.Conv1d(1, 1, kernel_size=self.k_max, padding=int(self.k_max/2), bias=False) 
        self.c_reparam.weight.data = kernal
        self.__delattr__('convc1')
        self.__delattr__('convc2')
        self.__delattr__('convc3')
        self.deploy = True



class SE(nn.Module):
    """ SE attention module in SENet [1].
        Args:
            channel: the number of channels
            reduction: a parameter to control model complexity
        Return:
            return a refined feature map
    """
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out
 
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)
 
 
class CBAM(nn.Module):
    """ CBAM attention module in CBAM [2].
        Args:
            channel: the number of channels
            reduction: a parameter to control model complexity
            kernel_size: 2-D convolution kernel size of the SAM module
        Return:
            return a refined feature map
    """
    def __init__(self, channel, reduction=16, kernel_size=7):  
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channel, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
 
    def forward(self, x):
        out = self.channel_attention(x) * x 
        out = self.spatial_attention(out) * out
        return out



class CA(nn.Module):
    """ CA attention module in CA [3].
        Args:
            channel: the number of channels
            reduction: a parameter to control model complexity
        Return:
            return a refined feature map
    """
    def __init__(self, channels,  reduction=32):
        super(CA, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = hswish()

        self.conv2 = nn.Conv2d(temp_c, channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h


class ECA(nn.Module):
     """Constructs a ECA module in ECANet [4].
     Args:
         channel: Number of channels of the input feature map
         k_size: Adaptive selection of kernel size
     Return:
            return a refined feature map
     """
     def __init__(self, channel, k_size=3):
         super(ECA, self).__init__()
         self.avg_pool = nn.AdaptiveAvgPool2d(1)
         self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
         self.sigmoid = nn.Sigmoid()

     def forward(self, x):
         # feature descriptor on the global spatial information
         y = self.avg_pool(x)

         # Two different branches of ECA module
         y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

         # Multi-scale information fusion
         y = self.sigmoid(y)
         x=x*y.expand_as(x)

         return x * y.expand_as(x)


class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = nn.Sequential(nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2),
                    nn.BatchNorm2d(1))
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

class TAM(nn.Module):
    """Constructs a TAM in TAM [5].
    Args:
         channel: Number of channels of the input feature map
         no_spatial: whether to use the spatial attention branch
    Return:
            return a refined feature map
     """
    def __init__(self, no_spatial=False):
        super(TAM, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out


class NAM(nn.Module):
    """Constructs the channel attention module of NAM  [6].
    Args:
         channel: Number of channels of the input feature map
    Return:
            return a refined feature map
    """
    def __init__(self, channels):
        super(NAM, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
 
 
    def forward(self, x):
        residual = x
        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual #
        
        return x
 



class SimAM(torch.nn.Module):
    """Constructs the module of SimAM  [7].
    Args:
         channel: Number of channels of the input feature map
    Return:
            return a refined feature map
    """
    def __init__(self, channels, e_lambda = 1e-4):
        super(SimAM, self).__init__()
 
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda
 
    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s
 
    @staticmethod
    def get_module_name():
        return "simam"
 
    def forward(self, x):
 
        b, c, h, w = x.size()
        
        n = w * h - 1
 
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
 
        return x * self.activaton(y)

"""

[1] Jie Hu, Li Shen, and Gang Sun. Squeeze-and-excitation net-
works. In Proceedings of the IEEE conference on computer
vision and pattern recognition, pages 7132–7141, 2018.
[2] Sanghyun Woo, Jongchan Park, Joon-Young Lee, and In So
Kweon. Cbam: Convolutional block attention module. In
Proceedings of the European conference on computer vision
(ECCV), pages 3–19, 2018.
[3] Qibin Hou, Daquan Zhou, and Jiashi Feng. Coordinate at-
tention for efficient mobile network design. In Proceedings
of the IEEE/CVF conference on computer vision and pattern
recognition, pages 13713–13722, 2021.
[4] Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wang-
meng Zuo, and Qinghua Hu. Eca-net: Efficient channel at-
tention for deep convolutional neural networks. In Proceed-
ings of the IEEE/CVF conference on computer vision and
pattern recognition, pages 11534–11542, 2020.
[5] Diganta Misra, Trikay Nalamada, Ajay Uppili Arasanipalai,
and Qibin Hou. Rotate to attend: Convolutional triplet atten-
tion module. In Proceedings of the IEEE/CVF Winter Con-
ference on Applications of Computer Vision, pages 3139–
3148, 2021.
[6] Yichao Liu, Zongru Shao, Yueyang Teng, and Nico Hoff-
mann. Nam: Normalization-based attention module. arXiv
preprint arXiv:2111.12419, 2021.
[7] Lingxiao Yang, Ru-Yuan Zhang, Lida Li, and Xiaohua Xie.
Simam: A simple, parameter-free attention module for con-
volutional neural networks. In International conference on
machine learning, pages 11863–11874. PMLR, 2021.

"""
