from torch import nn


class BasicBlock(nn.Module):
    """
    (b,c,y,x) -> (b,c,y,x)
    """
    expansion = 1

    def __init__(self, planes, bn_momentum=.1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=bn_momentum)

    def forward(self, x):
        residual = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += residual
        return self.relu(out)


if __name__ == '__main__':
    import torch

    model = BasicBlock(256)
    x = torch.randn(1, 256, 128, 128)
    print(model(x).size())  # torch.Size([1,256,128,128])
