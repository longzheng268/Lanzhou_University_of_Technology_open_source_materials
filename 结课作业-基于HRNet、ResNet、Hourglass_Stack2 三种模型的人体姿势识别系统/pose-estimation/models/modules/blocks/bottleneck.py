from torch import nn


class Bottleneck(nn.Module):
    """
    (b,c_in,y,x) -> (b,4*c_out,y,x)
    """

    expansion = 4

    def __init__(self, inplanes, planes, downsample=None, bn_momentum=.1):
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes,
                               kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=bn_momentum)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return self.relu(out)


if __name__ == '__main__':
    import torch

    downsample = nn.Sequential(
        nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
        nn.BatchNorm2d(256),
    )
    model = Bottleneck(64, 64, downsample=downsample)
    x = torch.randn(1, 64, 128, 128)
    print(model(x).size()) # torch.Size([1,256,128,128])

    model = Bottleneck(256,64)
    x = torch.randn(1,256,128,128)
    print(model(x).size()) # torch.Size([2,256,128,128])
