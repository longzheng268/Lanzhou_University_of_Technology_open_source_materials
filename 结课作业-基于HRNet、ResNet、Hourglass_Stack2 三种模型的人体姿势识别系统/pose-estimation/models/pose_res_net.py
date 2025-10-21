import torch
from torch import nn
from torchvision.models import resnet50
import torch.nn.functional as F


class PoseResNet(nn.Module):
    def __init__(self, keypoint_num=16):
        super(PoseResNet, self).__init__()

        # 加载预训练权重
        self.bottleneck = nn.Sequential(
            *list(resnet50(pretrained=True).children())[:-2])

        self.deconv_layer1 = nn.ConvTranspose2d(2048, 256, kernel_size=4,
                                                stride=2, padding=1, bias=False)
        self.de_bn1 = nn.BatchNorm2d(256)

        self.deconv_layer2 = nn.ConvTranspose2d(256, 256, kernel_size=4,
                                                stride=2, padding=1, bias=False)
        self.de_bn2 = nn.BatchNorm2d(256)

        self.deconv_layer3 = nn.ConvTranspose2d(256, 256, kernel_size=4,
                                                stride=2, padding=1, bias=False)
        self.de_bn3 = nn.BatchNorm2d(256)

        self.final_layer = nn.Conv2d(256, keypoint_num, kernel_size=1)

        nn.init.normal_(self.deconv_layer1.weight, std=.001)
        nn.init.normal_(self.deconv_layer2.weight, std=.001)
        nn.init.normal_(self.deconv_layer3.weight, std=.001)

        nn.init.constant_(self.de_bn1.weight, 1)
        nn.init.constant_(self.de_bn2.weight, 1)
        nn.init.constant_(self.de_bn3.weight, 1)
        nn.init.constant_(self.de_bn1.bias, 0)
        nn.init.constant_(self.de_bn2.bias, 0)
        nn.init.constant_(self.de_bn3.bias, 0)

        nn.init.normal_(self.final_layer.weight, std=.001)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(self, x):
        out = self.bottleneck(x)

        out = F.relu(self.de_bn1(self.deconv_layer1(out)))
        out = F.relu(self.de_bn2(self.deconv_layer2(out)))
        out = F.relu(self.de_bn3(self.deconv_layer3(out)))

        out = self.final_layer(out)

        return out


if __name__ == '__main__':
    model = PoseResNet()
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.size())
