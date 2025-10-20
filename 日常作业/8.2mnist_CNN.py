import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import datetime

# 设置随机种子
torch.manual_seed(0)

# 下载MNIST数据集并进行预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(32*12*12, 128)  # 输入特征数应根据卷积层输出调整
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.dropout(x)
        x = x.view(-1, 32*12*12)  # 将数据展平
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.softmax(self.fc2(x), dim=1)
        return x

model = CNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 计算每一层的参数数量
total_params = sum(p.numel() for p in model.parameters())
print("总共需要学习的参数数量: ", total_params)
print("每一层的参数数量:")
for i, (name, param) in enumerate(model.named_parameters()):
    print(f"Layer {i}: {name} - Params: {param.numel()}")

startdate = datetime.datetime.now()  # 获取当前时间

# 训练模型
for epoch in range(2):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

enddate = datetime.datetime.now()

print("训练用时：" + str(enddate - startdate))
