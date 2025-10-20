import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# 王少然210162401044
# 读取数据
data = pd.read_csv('ellipseSamples.txt', delim_whitespace=True, header=None)
X = data.iloc[:, :-1].values  # 特征
y = data.iloc[:, -1].values    # 标签

# 创建逻辑回归模型
model = LogisticRegression()

# 训练模型
model.fit(X, y)

# 创建网格以评估模型
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

# 增加步长以减少网格大小
step_size = 5.0  # 更大的步长，例如 5.0
xx, yy = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

# 预测每个网格点的类
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制决策边界
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
plt.title('Logistic Regression Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
