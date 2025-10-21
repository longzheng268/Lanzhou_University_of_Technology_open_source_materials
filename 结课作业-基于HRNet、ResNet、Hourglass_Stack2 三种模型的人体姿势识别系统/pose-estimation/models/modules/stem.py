from torch import nn


class Stem(nn.Module):
    """
    Stem模块进行1/4的下采样，并将通道数变为64
    (b,3,y,x) -> (b,64,y/4,x/4)
    """
    def __init__(self, bn_momentum=.1):
        super(Stem, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        return self.relu(out)

if __name__ == '__main__':
    import torch

    model = Stem()
    x = torch.randn(1,3,128,64)
    print(model(x).size()) # torch.Size([1,64,32,16])
