from .utils import IntermediateLayerGetter
from ._deeplab import (DeepLabHead,
                       DeepLabHeadV3Plus, 
                       DeepLabV3, 
                       )
from .backbone import resnet, tsm_resnet

from torch import nn
import torch
from torch.nn import functional as F

def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone, in_channels=3):
    print(f"Creating model with in_channels={in_channels}")
    if output_stride==8:
        replace_stride_with_dilation=[False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation=[False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)
    
    if in_channels != 3:
        print(f"Modifying first conv layer to accept {in_channels} channels")  # 添加調試信息
        original_conv = backbone.conv1
        backbone.conv1 = nn.Conv2d(
            in_channels,
            original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        if pretrained_backbone:
            print("Initializing new conv layer with pretrained weights")  # 添加調試信息
            with torch.no_grad():
                backbone.conv1.weight[:, :3, :, :].data.copy_(original_conv.weight.data)
                original_weight_mean = original_conv.weight.mean(dim=1, keepdim=True)
                for i in range(3, in_channels):
                    backbone.conv1.weight[:, i:i+1, :, :].data.copy_(original_weight_mean)
    inplanes = 2048
    low_level_planes = 256

    if name=='deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name=='deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes , num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model

def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone, temporal=False, model_type='parallel', opts=None, in_channels=3):
    if backbone.startswith('resnet'):
        # 創建基本模型
        base_model = _segm_resnet(
            arch_type,
            backbone,
            num_classes,
            output_stride=output_stride,
            pretrained_backbone=pretrained_backbone,
            in_channels=in_channels
        )
        model = base_model
    else:
        raise NotImplementedError
    
    return model

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride, pretrained_backbone=pretrained_backbone)