# -*- coding: utf-8 -*-
from jieba import lcut
from gensim.similarities import SparseMatrixSimilarity
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
# 文本集和搜索词
texts = ['#宽窄观察[超话]# 非常时刻，安倍访华，贸易摩擦出现最大变数…非常时刻，安倍访华，贸易摩擦出现最大变数 http://t.cn/EZxq5Qy',
         '#宽窄观察# 【这个情形会不会是贸易战的结果和隐喻？】在常人眼里，政治就是听话，服从上级，其实错了。政治，不是服从，而是妥协。妥协是政治最基本的内涵。而妥协，本身又是一门艺术。如何妥协，何时妥协，向谁妥协，是非常讲究的。这涉及到对天时、地利、人和的理解和掌握。 妥协的背后是什么？是实力。没有实力的支撑，你就没有资格要求人家与你妥协。#反击贸易战# http://t.cn/Rndh0X6',
         '#宽窄思语# 继续说房子。80后90后是幸福的童年，悲催的成年，房地产、教育、医疗公共产品领域市场化的结果，变成了他们身上的三座大山。不必说买不起房子的，勉强为之交了首付的立马成为房奴。现在贸易战一起来，三架马车的出口受阻，好多人工作难保，房贷呢？交不交？买了还装不装？这个背景下说提振内需就是一厢情愿。一个社会，年轻人看不到未来，这个社会还有未来吗？日本迷失二十年后的低欲望社会，正在被我们复制。于是，继佛系青年、养生热潮之后，第一批90后已经悄然开始“消费降级”了。http://t.cn/RuliyF5']
keyword = '#宽窄观察# 大豆上场了！[二哈][二哈]#反击贸易战#'
# 1、将【文本集】生成【分词列表】
texts = [lcut(text) for text in texts]
# print(texts)
# 2、基于文本集建立【词典】，并获得词典特征数
dictionary = Dictionary(texts)
# for i, j in dictionary.items():
#     print(i, j)

# print(dictionary.dfs)  # 词频
# dictionary.filter_n_most_frequent(1)
# print(dictionary.dfs)  # 词频
num_features = len(dictionary.token2id)
# 3.1、基于词典，将【分词列表集】转换成【稀疏向量集】，称作【语料库】
# print(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# for text in texts:
#     print([dictionary.doc2bow(text)])
#     print(text)
# print(corpus)

# 3.2、同理，用【词典】把【搜索词】也转换为【稀疏向量】
kw_vector = dictionary.doc2bow(lcut(keyword))
# 4、创建【TF-IDF模型】，传入【语料库】来训练
tfidf = TfidfModel(corpus)
# print(tfidf)
# 5、用训练好的【TF-IDF模型】处理【被检索文本】和【搜索词】
tf_texts = tfidf[corpus]  # 此处将【语料库】用作【被检索文本】
tf_kw = tfidf[kw_vector]
for i in tfidf[corpus]:
    print(i)
# 6、相似度计算
sparse_matrix = SparseMatrixSimilarity(tf_texts, num_features)

similarities = sparse_matrix.get_similarities(tf_kw)
for e, s in enumerate(similarities, 1):
    print('keyword 与 text%d 相似度为：%.2f' % (e, s))
