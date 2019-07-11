# coding=utf-8
"""
Word2vec 文本聚类
三种聚类算法：Kmeans DBSCAN GMM（其中GMM很慢）
200维词向量（从维基百科1.3G中文数据训练得到）
输入：prepro.py 形成的pkl文件
输出：聚类图形
      每一类的每条微博内容 ："cluster1.txt" 、"cluster2.txt" 、"cluster3.txt" 、"cluster4.txt"
参数：weight权重 这是一个重要参数
"""
import os
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence


import time
import re
import os
import sys
import codecs
import pickle
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from matplotlib.font_manager import *

from sklearn.decomposition import PCA
import shutil
import numpy as np
import matplotlib
import scipy
import matplotlib.pyplot as plt
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


maxlen = 200

def input_transform(tokenize):

    # print(words.shape)
    words = np.array(tokenize).reshape(1, -1)
    print(words.shape)
    model = Word2Vec.load('../Sentiment-Analysis-master/lstm_data/wiki.zh.text.model')
    # 举例子，相近词

    # for key in model.similar_by_word(u'日本', topn=10):
    #     # if len(key[0]) == 3:  # key[0]应该就表示某个词
    #     print(key[0], key[1])  # 某一个词,某一个词出现的概率

    _, _, combined = create_dictionaries(model, words)
    return combined

def create_dictionaries(model=None, combined=None):
    '''
        Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries
    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab,
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()}  # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()}  # 所有频数超过10的词语的词向量

        def parse_dataset(combined):
            ''' Words become integers
            '''
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data
        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        # 每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


if __name__ == "__main__":

    #########################################################################
    #                           第一步 计算DOC2VEC

    # 文档预料 空格连接

    weight = []
    content_comment = pickle.load(open('./Agu.pkl', 'rb'))

    # 读取预料 一行预料为一个文档
    for i in content_comment:
        word_vec = input_transform(np.array(i[2]).reshape(1, -1))
        weight.append(word_vec[0])
    # print(corpus)

    # 参考: http://blog.csdn.net/abcjennifer/article/details/23615947
    # vectorizer = HashingVectorizer(n_features = 4000)

    # 将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频

    weight = np.array(weight)
    print("weight...")  # list of list格式
    print(weight[200:])

    ########################################################################
    #                               第二步 聚类Kmeans

    print('Start Kmeans:')

    clf = KMeans(n_clusters=3)  # 景区 动物 人物 国家
    s = clf.fit(weight)
    print(s)

    # print 'Start MiniBatchKmeans:'
    # from sklearn.cluster import MiniBatchKMeans
    # clf = MiniBatchKMeans(n_clusters=20)
    # s = clf.fit(weight)
    # print s

    # 中心点
    # print(clf.cluster_centers_)

    # 每个样本所属的簇
    label = []  # 存储1000个类标 4个类
    print(clf.labels_)
    i = 1
    while i <= len(clf.labels_):
        print(i, clf.labels_[i - 1])
        label.append(clf.labels_[i - 1])
        i = i + 1

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数  958.137281791
    print(clf.inertia_)


    ########################################################################
    #                               第三步 图形输出 降维

    pca = PCA(n_components=2)  # 输出两维
    newData = pca.fit_transform(weight)  # 载入N维
    print(newData)
    print(len(newData))

    x1 = []
    y1 = []

    x2 = []
    y2 = []

    x3 = []
    y3 = []

    x4 = []
    y4 = []

    for index, value in enumerate(label):
        if value == 0:
            x1.append(newData[index][0])
            y1.append(newData[index][1])
        elif value == 1:
            x2.append(newData[index][0])
            y2.append(newData[index][1])
        elif value == 2:
            x3.append(newData[index][0])
            y3.append(newData[index][1])
        elif value == 3:
            x4.append(newData[index][0])
            y4.append(newData[index][1])

    # 四种颜色 红 绿 蓝 黑
    path = os.path.join(os.path.dirname(__file__), 'cluster_2')
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    plt.rcParams['savefig.dpi'] = 300 #图片像素
    plt.rcParams['figure.dpi'] = 300#分辨率
    plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
    # 服务器上貌似还是不能很好地显示中文，于是：
    myfont = FontProperties(fname='../msyh.ttc')
    plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
    plt.plot(x1, y1, 'or')
    plt.plot(x2, y2, 'og')
    plt.plot(x3, y3, 'ob')
    plt.plot(x4, y4, 'ok')
    plt.title('词向量聚类效果', fontproperties=myfont)
    plt.savefig('cluster_w2v',dpi=300)
    plt.show()
    

    ########################################################################
    #                        第四步 文本聚类结果写入文件
    path = os.path.join('../', 'cluster_2')
    if not os.path.exists(path):
        os.mkdir(path)
    os.chdir(path)
    clustername1 = "cluster1.txt"
    clustername2 = "cluster2.txt"
    clustername3 = "cluster3.txt"
    clustername4 = "cluster4.txt"
    cluster1 = codecs.open(clustername1, 'w', 'utf-8')
    cluster2 = codecs.open(clustername2, 'w', 'utf-8')
    cluster3 = codecs.open(clustername3, 'w', 'utf-8')
    cluster4 = codecs.open(clustername4, 'w', 'utf-8')

    for index, value in enumerate(label):
        print(content_comment[index][1])
        if value == 0:
            cluster1.write(content_comment[index][1] + '\n')
            cluster1.write(' ' + '\n')
        elif value == 1:
            cluster2.write(content_comment[index][1] + '\n')
            cluster2.write(' ' + '\n')
        elif value == 2:
            cluster3.write(content_comment[index][1] + '\n')
            cluster3.write(' ' + '\n')
        elif value == 3:
            cluster4.write(content_comment[index][1] + '\n')
            cluster4.write(' ' + '\n')

    cluster1.close()
    cluster2.close()
    cluster3.close()
    cluster4.close()


