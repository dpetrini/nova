
# Resnet50

import torch
from torchvision import models
from torch.hub import load_state_dict_from_url


class MyResnet50(models.ResNet):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):

        # Start with standard resnet18 defined here
        super().__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3],
                         num_classes=num_classes, **kwargs)
        if pretrained:
            state_dict = load_state_dict_from_url(
                          models.resnet.model_urls["resnet50"],
                          progress=True)
            self.load_state_dict(state_dict)

        print("HEREi 1")

    # Reimplementing forward pass
    def _forward_impl(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        #print('0: ', x.shape)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        #print('1: ', x.shape)
        # x = nn.AdaptiveAvgPool2d(x)

        x = torch.flatten(x, 1)   # comentar para rodar mostra heatmap
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)
