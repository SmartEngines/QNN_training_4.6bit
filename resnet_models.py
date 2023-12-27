from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from quantized_layers import FConv2d, QConv2d

from torchvision.models import resnet18, resnet34, ResNet18_Weights, ResNet34_Weights


class MyQBasicBlock(nn.Module):
    # refactured and simplified version of ResNet's BasicBlock from Tochvision library
    # see: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = FConv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False, batch_norm=True,
                             pre_activation=F.relu6)
        self.conv2 = FConv2d(planes, planes, 3, stride=1, padding=1, bias=False, batch_norm=True,
                             pre_activation=F.relu6)
        self.downsample = downsample
        self.relu = nn.ReLU6(inplace=True)
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class MyQResNet(nn.Module):
    # refactured and simplified version of ResNet from Tochvision library
    # see: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000
    ) -> None:
        super().__init__()

        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = FConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = FConv2d(
                self.inplanes, planes, 1, stride, bias=False, batch_norm=True, pre_activation=F.relu6
            )

        layers = []

        layers.append(
            MyQBasicBlock(
                self.inplanes, planes, stride, downsample
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                MyQBasicBlock(
                    self.inplanes,
                    planes
                )
            )

        return nn.Sequential(*layers)


    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_qresnet_18(pretrained=True, device='cpu'):
    model = MyQResNet([2, 2, 2, 2]).to(device)
    if pretrained:
        original_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        paired_layers = [
            (model.conv1.conv, original_model.conv1),
            (model.conv1.bn,   original_model.bn1),
            (model.layer1[0].conv1.conv, original_model.layer1[0].conv1),
            (model.layer1[0].conv1.bn,   original_model.layer1[0].bn1),
            (model.layer1[0].conv2.conv, original_model.layer1[0].conv2),
            (model.layer1[0].conv2.bn,   original_model.layer1[0].bn2),
            (model.layer1[1].conv1.conv, original_model.layer1[1].conv1),
            (model.layer1[1].conv1.bn,   original_model.layer1[1].bn1),
            (model.layer1[1].conv2.conv, original_model.layer1[1].conv2),
            (model.layer1[1].conv2.bn,   original_model.layer1[1].bn2),

            (model.layer2[0].conv1.conv, original_model.layer2[0].conv1),
            (model.layer2[0].conv1.bn,   original_model.layer2[0].bn1),
            (model.layer2[0].conv2.conv, original_model.layer2[0].conv2),
            (model.layer2[0].conv2.bn,   original_model.layer2[0].bn2),
            (model.layer2[0].downsample.conv, original_model.layer2[0].downsample[0]),
            (model.layer2[0].downsample.bn,   original_model.layer2[0].downsample[1]),
            (model.layer2[1].conv1.conv, original_model.layer2[1].conv1),
            (model.layer2[1].conv1.bn,   original_model.layer2[1].bn1),
            (model.layer2[1].conv2.conv, original_model.layer2[1].conv2),
            (model.layer2[1].conv2.bn,   original_model.layer2[1].bn2),

            (model.layer3[0].conv1.conv, original_model.layer3[0].conv1),
            (model.layer3[0].conv1.bn,   original_model.layer3[0].bn1),
            (model.layer3[0].conv2.conv, original_model.layer3[0].conv2),
            (model.layer3[0].conv2.bn,   original_model.layer3[0].bn2),
            (model.layer3[0].downsample.conv, original_model.layer3[0].downsample[0]),
            (model.layer3[0].downsample.bn,   original_model.layer3[0].downsample[1]),
            (model.layer3[1].conv1.conv, original_model.layer3[1].conv1),
            (model.layer3[1].conv1.bn,   original_model.layer3[1].bn1),
            (model.layer3[1].conv2.conv, original_model.layer3[1].conv2),
            (model.layer3[1].conv2.bn,   original_model.layer3[1].bn2),

            (model.layer4[0].conv1.conv, original_model.layer4[0].conv1),
            (model.layer4[0].conv1.bn,   original_model.layer4[0].bn1),
            (model.layer4[0].conv2.conv, original_model.layer4[0].conv2),
            (model.layer4[0].conv2.bn,   original_model.layer4[0].bn2),
            (model.layer4[0].downsample.conv, original_model.layer4[0].downsample[0]),
            (model.layer4[0].downsample.bn,   original_model.layer4[0].downsample[1]),
            (model.layer4[1].conv1.conv, original_model.layer4[1].conv1),
            (model.layer4[1].conv1.bn,   original_model.layer4[1].bn1),
            (model.layer4[1].conv2.conv, original_model.layer4[1].conv2),
            (model.layer4[1].conv2.bn,   original_model.layer4[1].bn2),

            (model.fc,   original_model.fc)
        ]

        for new_layer, old_layer in paired_layers:
            new_layer.load_state_dict(old_layer.state_dict())
        del original_model
    return model
