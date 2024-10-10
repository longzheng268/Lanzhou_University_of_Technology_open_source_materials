import nltk

# 下载常用的 NLTK 数据集和模型
nltk.download('punkt')     # 用于文本分词
nltk.download('averaged_perceptron_tagger')  # 常用词性标注器

# 输入句子
text = "Natural Language Processing is a fascinating field."

# 对句子进行分词
words = nltk.word_tokenize(text)

# 进行词性标注
pos_tags = nltk.pos_tag(words)

# 输出结果
print("Tokenized Words:", words)
print("POS Tags:", pos_tags)
# 使用 PerceptronTagger 进行标注
from nltk.tag import PerceptronTagger

tagger = PerceptronTagger()
pos_tags_perceptron = tagger.tag(words)

print("Perceptron POS Tags:", pos_tags_perceptron)
