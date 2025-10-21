import torch
from torch import nn

from models.modules.blocks.bottleneck import Bottleneck
from models.modules.stage_module import StageModule


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            

class HRNet(nn.Module):

    def __init__(self, c=48, nof_joints=16, bn_momentum=.1):
        super(HRNet, self).__init__()

        # (b,3,y,x) -> (b,64,y,x)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3,
                               stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=bn_momentum)
        self.relu = nn.ReLU(inplace=True)

        # (b,64,y,x) -> (b,256,y,x)
        downsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )
        self.layer1 = nn.Sequential(
            Bottleneck(64, 64, downsample=downsample),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
            Bottleneck(256, 64),
        )

        # (b,256,y,x) ---+---> (b,c,y,x)
        #                +---> (b,c*2,y/2,x/2)
        self.transition1 = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, c, kernel_size=3,
                          stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(256, c * 2, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 2),
                nn.ReLU(inplace=True),
            ))
        ])

        # StageModule中每个分枝发生了融合
        # (b,c,y,x) ------+---> (b,c,y,x)
        # (b,c*2,y/2,x/2) +---> (b,c*2,y/2,x/2)
        self.stage2 = nn.Sequential(
            StageModule(stage=2, output_branches=2, c=c, bn_momentum=bn_momentum)
        )

        # (b,c,y,x) ----------> (b,c,y,x)
        # (b,c*2,y/2,x/2) +---> (b,c*2,y/2,x/2)
        #                 +---> (b,c*4,y/4,x/4)
        self.transition2 = nn.ModuleList([
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(nn.Sequential(
                nn.Conv2d(c * 2, c * 4, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 4),
                nn.ReLU(inplace=True),
            ))
        ])

        # (b,c,y,x) ------++++---> (b,c,y,x)
        # (b,c*2,y/2,x/2) ++++---> (b,c*2,y/2,x/2)
        # (b,c*4,y/4,x/4) ++++---> (b,c*4,y/4,x/4)
        self.stage3 = nn.Sequential(
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
            StageModule(stage=3, output_branches=3, c=c, bn_momentum=bn_momentum),
        )

        # (b,c,y,x) ----------> (b,c,y,x)
        # (b,c*2,y/2,x/2) ----> (b,c*2,y/2,x/2)
        # (b,c*4,y/4,x/4) +---> (b,c*4,y/4,x/4)
        #                 +---> (b,c*8,y/8,x/8)
        self.transition3 = nn.ModuleList([
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(),  # None,   - Used in place of "None" because it is callable
            nn.Sequential(nn.Sequential(  # Double Sequential to fit with official pretrained weights
                nn.Conv2d(c * 4, c * 8, kernel_size=3,
                          stride=2, padding=1, bias=False),
                nn.BatchNorm2d(c * 8),
                nn.ReLU(inplace=True),
            )),
        ])

        # (b,c,y,x) ------+++---> (b,c,y,x)
        # (b,c*2,y/2,x/2) +++---> (b,c*2,y/2,x/2)
        # (b,c*4,y/4,x/4) +++---> (b,c*4,y/4,x/4)
        # (b,c*8,y/8,x/8) +++---> (b,c*8,y/8,x/8)
        self.stage4 = nn.Sequential(
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=4, c=c, bn_momentum=bn_momentum),
            StageModule(stage=4, output_branches=1, c=c, bn_momentum=bn_momentum),
        )

        # 取最高分辨率的结果
        # (b,c,y,x) -> (b,nof_joints*2,y,x)
        self.final_layer = nn.Conv2d(c, nof_joints, kernel_size=1, stride=1)

        self.apply(weights_init)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        x = self.layer1(x)
        x = [trans(x) for trans in self.transition1]

        x = self.stage2(x)
        x = [
            self.transition2[0](x[0]),
            self.transition2[1](x[1]),
            self.transition2[2](x[1]),
        ]

        x = self.stage3(x)
        x = [
            self.transition3[0](x[0]),
            self.transition3[1](x[1]),
            self.transition3[2](x[2]),
            self.transition3[3](x[2]),
        ]

        x = self.stage4(x)

        x = x[0]
        out = self.final_layer(x)

        return out

def hr_w32():
    return HRNet(32)

if __name__ == '__main__':
    import torch

    model = hr_w32()
    x = torch.randn(1,3,256,256)
    output = model(x)
    print(output.size())
