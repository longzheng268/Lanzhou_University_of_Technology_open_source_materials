import nltk
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

# 下载所需的 NLTK 数据集
nltk.download('conll2002')

# 加载 CoNLL-2003 数据集的西班牙语部分
from nltk.corpus import conll2002
train_data = conll2002.iob_sents('esp.train')
test_data = conll2002.iob_sents('esp.testb')

# 定义特征提取函数
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }

    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True  # 句子开始

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True  # 句子结束

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

# 提取训练集和测试集的特征和标签
X_train = [sent2features(s) for s in train_data]
y_train = [sent2labels(s) for s in train_data]

X_test = [sent2features(s) for s in test_data]
y_test = [sent2labels(s) for s in test_data]

# 初始化 CRF 模型
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    c1=0.1,  # L1 正则化系数
    c2=0.1,  # L2 正则化系数
    max_iterations=100,
    all_possible_transitions=True
)

# 训练 CRF 模型
print("Training CRF model...")
crf.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = crf.predict(X_test)

# 使用 F1-score 进行评估
labels = list(crf.classes_)
labels.remove('O')  # 'O' 代表非命名实体
f1_score = metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)

print(f"F1 score: {f1_score}")

# 打印分类报告
report = metrics.flat_classification_report(
    y_test, y_pred, labels=labels, digits=3
)
print(report)

# 超参数调优 - 网格搜索
params_space = {
    'c1': [0.01, 0.1, 1.0],
    'c2': [0.01, 0.1, 1.0]
}

# 使用随机搜索进行超参数选择
rs = RandomizedSearchCV(crf, params_space,
                        cv=3,
                        verbose=1,
                        n_jobs=-1,
                        scoring='f1_weighted')

print("Performing hyperparameter tuning...")
rs.fit(X_train, y_train)

# 输出最佳参数
print(f"Best parameters found: {rs.best_params_}")
