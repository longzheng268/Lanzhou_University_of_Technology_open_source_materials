import numpy as np
import pickle
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

def load_cifar10_data(data_dir):
    x_train, y_train = [], []
    x_test, y_test = [], []

    # 加载训练数据
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        print(f"Loading {batch_file}")
        with open(batch_file, 'rb') as file:
            batch = pickle.load(file, encoding='latin1')
            x_train.append(batch['data'])
            y_train += batch['labels']

    x_train = np.concatenate(x_train)
    x_train = x_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    y_train = to_categorical(y_train, num_classes=10)

    # 加载测试数据
    test_batch_file = os.path.join(data_dir, 'test_batch')
    print(f"Loading {test_batch_file}")
    with open(test_batch_file, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
        x_test = batch['data']
        y_test = batch['labels']

    x_test = x_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32') / 255.0
    y_test = to_categorical(y_test, num_classes=10)

    return (x_train, y_train), (x_test, y_test)

# 设置数据目录
data_dir = r'D:\系统默认\文档\Tencent Files\2264162324\FileRecv\MobileFile\cifar-10-python\cifar-10-batches-py'

# 加载数据
try:
    (x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)
    print(f'Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}')
    print(f'Testing data shape: {x_test.shape}, Testing labels shape: {y_test.shape}')
except FileNotFoundError as e:
    print(e)

# 构建卷积神经网络模型
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # 第一层卷积层
model.add(layers.MaxPooling2D((2, 2)))  # 池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 第二层卷积层
model.add(layers.MaxPooling2D((2, 2)))  # 池化层
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 第三层卷积层
model.add(layers.Flatten())  # 展平层
model.add(layers.Dense(64, activation='relu'))  # 全连接层
model.add(layers.Dense(10, activation='softmax'))  # 输出层，10类分类

# 编译模型
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.1)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')
