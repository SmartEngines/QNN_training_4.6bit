import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
import torchvision.transforms as transforms


from quantized_layers import FConv2d, QConv2d
from typing import List, Optional


resnet_transformer_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

resnet_transformer_test = transforms.Compose([
    transforms.ToTensor(),
    torchvision.models.ResNet18_Weights.IMAGENET1K_V1.transforms(),
])


class MyQBasicBlock(nn.Module):
    # refactured and simplified version of ResNet's BasicBlock from Tochvision library
    # see: https://pytorch.org/vision/stable/_modules/torchvision/models/resnet.html
    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            quantized: bool = False
    ) -> None:
        super().__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = FConv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False, batch_norm=True,
                             pre_activation=F.relu6)
        self.conv2 = FConv2d(planes, planes, 3, stride=1, padding=1, bias=False, batch_norm=True,
                             pre_activation=F.relu6)
        if quantized:
            self.conv1 = QConv2d(self.conv1, quant_mode='void')
            self.conv2 = QConv2d(self.conv2, quant_mode='void')
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
            num_classes: int = 1000,
            quantized=False
    ) -> None:
        super().__init__()
        self.quantized = quantized

        self.groups = 1
        self.base_width = 64
        self.inplanes = 64
        self.dilation = 1
        self.conv1 = FConv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU6(inplace=True)
        self.maxpool = nn.MaxPool2d(2)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
            if self.quantized:
                downsample = QConv2d(downsample, quant_mode='void')

        layers = []

        layers.append(
            MyQBasicBlock(
                self.inplanes, planes, stride, downsample, quantized=self.quantized
            )
        )
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(
                MyQBasicBlock(
                    self.inplanes,
                    planes,
                    quantized=self.quantized
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

