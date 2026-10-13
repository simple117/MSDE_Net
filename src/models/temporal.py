import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbones.resnet import resnet18
from src.models.backbones.vgg import vgg16, vgg16_bn
from src.models.backbones import c3d
from src.models.backbones.r3d import r3d_18, r2plus1d_18
from src.models.attention import EncoderSelfAttention
import torch_dct as dct
# from src.models.cross_fusion import CrossModalFusion
# from src.models.cross_fusion import CrossModalFusionWithCGAF_SelfAttn
# from src.models.cross_fusion4 import CrossModalFusionWithCGAF_SelfAttn
from src.models.cross_fusion2 import CrossModalFusionWithCGAF_SelfAttn
# from src.performer_t11 import GlobalTemporalModel
# from src.performer_t10 import GlobalTemporalModel
from src.performer_t9 import GlobalTemporalModel

backbone_dict = {'resnet': resnet18,
                 'vgg': vgg16, 'vgg_bn': vgg16_bn,
                 'c3d': c3d,
                 'r3d': r3d_18, 'r2plus1d': r2plus1d_18}


def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage



class _GestureTransformer(nn.Module):
    """Multi Modal model for gesture recognition on 3 channel"""
    def __init__(self, backbone: nn.Module, in_planes: int, out_planes: int,
                 pretrained: bool = False, dropout_backbone=0.1,
                 **kwargs):
        super(_GestureTransformer, self).__init__()

        self.in_planes = in_planes
        self.backbone = backbone(pretrained, in_planes, dropout=dropout_backbone)

        # self.self_attention = EncoderSelfAttention(512, 64, 64, **kwargs)

        # self.cross_fusion = CrossModalFusionWithCGAF_SelfAttn()

        self.cross_fusion=CrossModalFusionWithCGAF_SelfAttn(
        embed_dim=512, num_heads=8, num_cgaf_stages=2,
        ffn_dim_multiplier=4, output_dim=512, dropout=0.1
    )

        # —— 3) 全局时序聚合 —— #
        # 这里用 MSTCN 构成的 GlobalTemporalModel
        embed_dim=512
        global_blocks=2
        self.global_model = GlobalTemporalModel(dim=embed_dim,
                                                num_blocks=global_blocks,
                                                kernel_sizes=[3,5,7])
        self.classifier = nn.Linear(512, out_planes)


    def forward(self, x,y):
        shape = x.shape

        x = x.view(-1, self.in_planes, x.shape[-2], x.shape[-1])

        # 魔改
        x= self.backbone(x)
        x = x.view(shape[0], shape[1] // self.in_planes, -1)

        # y=torch.tensor(y,dtype=torch.float16)
        y=dct.dct(y)

        x=self.cross_fusion(x,y)

        # x = self.self_attention(x)
        x=self.global_model(x)

        x = self.classifier(x)
        return x

def GestureTransoformer(backbone: str="resnet", in_planes: int=3, n_classes: int=25, **kwargs):
    if backbone not in backbone_dict:
        raise NotImplementedError("Backbone type: [{}] is not implemented.".format(backbone))
    model = _GestureTransformer(backbone_dict[backbone], in_planes, n_classes, **kwargs)
    return model