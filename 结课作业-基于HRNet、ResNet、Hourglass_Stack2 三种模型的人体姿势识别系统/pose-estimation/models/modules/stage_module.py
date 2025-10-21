from torch import nn

from models.modules.blocks.basic_block import BasicBlock


class StageModule(nn.Module):

    def __init__(self, stage, output_branches, c, bn_momentum):
        super(StageModule, self).__init__()

        self.stage = stage
        self.output_branches = output_branches

        # 得到stage对应数量的分枝
        # 例如stage=3，c=32时
        # i = 0,1,2
        # i = 0 -> 4*BasicBlock(32)
        # i = 1 -> 4*BasicBlock(64)
        # i = 2 -> 4*BasicBlock(128)
        #
        # -+--- 4*BasicBlock(32) ---->
        #  +--- 4*BasicBlock(64) ---->
        #  +--- 4*BasicBlock(128) --->
        self.branches = nn.ModuleList()
        for i in range(self.stage):
            w = c * (2**i)
            branch = nn.Sequential(
                BasicBlock(w, bn_momentum=bn_momentum),
                BasicBlock(w, bn_momentum=bn_momentum),
                BasicBlock(w, bn_momentum=bn_momentum),
                BasicBlock(w, bn_momentum=bn_momentum),
            )
            self.branches.append(branch)

        self.fuse_layers = nn.ModuleList()

        # 得到i*j个输出分枝，其中第(i,j)个输出分枝代表第j个分枝向第i个输出变换的输出分枝
        # i<j，则输出分枝的通道数小于分枝i的通道数，作上采样
        # i>j，则输出分枝的通道数大于分枝i的通道数，作下采样
        #                     +---output branch 0(c=32)---->
        #                     +(upsample)
        # ---branch 1(c=64)---+---output branch 1(c=64)---->
        #                     +(downsample)
        #                     +---output branch 2(c=128)--->
        # 对于每一个输出分枝i
        for i in range(self.output_branches):
            self.fuse_layers.append(nn.ModuleList())

            # 对于每一个分枝j
            for j in range(self.stage):

                # 如果分枝与输出分枝相对应，直接输出
                if i == j:
                    self.fuse_layers[-1].append(nn.Sequential())

                # 如果输出分枝编号小于分枝编号，则上采样后输出
                elif i < j:
                    self.fuse_layers[-1].append(nn.Sequential(
                        nn.Conv2d(c * (2**j), c * (2**i), kernel_size=1,
                                  stride=1, bias=False),
                        nn.BatchNorm2d(c * (2**i)),
                        nn.Upsample(scale_factor=(2.**(j-i))),
                    ))

                # 如果输出分枝编号大于分枝编号，则下采样后输出
                elif i > j:
                    ops = []
                    for _ in range(i - j - 1):
                        ops.append(nn.Sequential(
                            nn.Conv2d(c * (2**j), c * (2**j), kernel_size=3,
                                      stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(c * (2**j)),
                            nn.ReLU(inplace=True),
                        ))
                    ops.append(nn.Sequential(
                        nn.Conv2d(c * (2**j), c * (2**i), kernel_size=3,
                                  stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(c * (2**i)),
                    ))
                    self.fuse_layers[-1].append(nn.Sequential(*ops))

            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 将x经过每个分枝
        x = [branch(b) for branch, b in zip(self.branches, x)]

        x_fused = []
        # 对于每个输出分枝
        for i in range(len(self.fuse_layers)):
            # 对于每个分枝
            for j in range(len(self.branches)):
                # 如果是第0个分枝，则将经过第0个分枝的x经过第i个输出分枝
                if j == 0:
                    x_fused.append(self.fuse_layers[i][0](x[0]))
                # 否则，将经过第j个分枝的x经过第i个输出分枝，与之前第i个输出分枝的结果相加
                else:
                    x_fused[i] = x_fused[i] + self.fuse_layers[i][j](x[j])

        # 每个输出分枝的结果经过ReLU
        for i in range(len(x_fused)):
            x_fused[i] = self.relu(x_fused[i])

        return x_fused
