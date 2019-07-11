# -*- coding: utf-8 -*-
'''
基于word2vec和LSTM神经网络进行情感分析
由于参考的原程序写的可读性有点差并且是早期python2.0版本，由于时间所限，故这里仅仅把程序跑通，并且简单说一下程序大致思路：
程序大致分为两步：
1.训练
训练主要基于train()函数，训练完保存LSTM的结构和权重参数
2.预测
预测主要基于lstm_predict()函数，将每条微博内容先转为w2v形式，然后再输入到LSTM中，最后得到二分类情感正负作为输出结果
会用到词向量预训练模型和LSTM预训练模型

需要以下文件：（部分文件太大看百度网盘的链接把）
1. 打完情感正负标签的训练集数据，用来作为训练集训练LSTM的参数
'../Sentiment-Analysis-master/data/neg.xls'
'../Sentiment-Analysis-master/data/pos.xls'
2. LSTM神经网络训练完得到的神经网络结构和权重参数
'../lstm_data/lstm.yml'
'../lstm_data/lstm.h5'
3. 基于维基百科1.3G繁体中文转简体训练的词向量模型
'../Sentiment-Analysis-master/lstm_data/wiki.zh.text.model'

'''
import yaml
import sys
# from smart_open import open
import pickle

from sklearn.model_selection import train_test_split
import multiprocessing
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
# import word2vec
from gensim.corpora.dictionary import Dictionary

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import jieba
import csv
import pandas as pd
import sys
sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 200
maxlen = 200
n_iterations = 1  # ideally more..
n_exposures = 5
window_size = 5
batch_size = 32
n_epoch = 4
input_length = 200
cpu_count = multiprocessing.cpu_count()


# 加载训练文件
def loadfile():
    neg = pd.read_excel('../../Sentiment-Analysis-master/data/neg.xls', header=None, index=None)
    pos = pd.read_excel('../../Sentiment-Analysis-master/data/pos.xls', header=None, index=None)

    combined = np.concatenate((pos[0], neg[0]))
    # print(combined)
    # print(combined.shape)
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neg), dtype=int)))

    return combined, y


# 对句子经行分词，并去掉换行符
def tokenizer(text):
    '''

    :param text:  string
    :return:  [[tokenize1,tokenize2..]] list of list
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
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
            '''
            Words become integers
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


# 创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec.load('../../Sentiment-Analysis-master/lstm_data/wiki.zh.text.model')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    print("combined, word2vec_train")
    print(combined)
    print(combined.shape)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))  # 索引为0的词语，词向量全为0
    for word, index in index_dict.items():  # 从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train.shape, y_train.shape)
    return n_symbols, embedding_weights, x_train, y_train, x_test, y_test


# 定义网络结构
def train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test):
    print('Defining a Simple Keras Model...')
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))  # Adding Input Length
    model.add(LSTM(activation="sigmoid", units=50, recurrent_activation="hard_sigmoid"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    print('Compiling the Model...')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print("Train...")
    print(x_train.shape)
    print(x_train)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('../lstm_data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('../lstm_data/lstm.h5')
    print('Test score:', score)


# 训练模型，并保存
def train():
    print('Loading Data...')
    combined, y = loadfile()
    print(len(combined), len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols, embedding_weights, x_train, y_train, x_test, y_test = get_data(index_dict, word_vectors, combined, y)
    print(x_train.shape, y_train.shape)
    train_lstm(n_symbols, embedding_weights, x_train, y_train, x_test, y_test)


def input_transform(string):
    words = jieba.lcut(string)
    # print(words.shape)
    words = np.array(words).reshape(1, -1)
    # print(words.shape)
    model = Word2Vec.load('../../Sentiment-Analysis-master/lstm_data/wiki.zh.text.model')
    # 举例子，相近词

    # for key in model.similar_by_word(u'日本', topn=10):
    #     # if len(key[0]) == 3:  # key[0]应该就表示某个词
    #     print(key[0], key[1])  # 某一个词,某一个词出现的概率

    _, _, combined = create_dictionaries(model, words)
    return combined

def lstm_predict(string):
    print('loading model......')
    with open('../../Sentiment-Analysis-master/lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print('loading weights......')
    model.load_weights('../../Sentiment-Analysis-master/lstm_data/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print('Word2Vec......')
    data = input_transform(string)
    # print(data)
    data.reshape(1, -1)
    # print(data.shape)
    # print(data)
    # print(len(data[0]))
    result = model.predict_classes(data)
    if result[0][0] == 1:
        # print(string, ' positive')
        return 'positive'
    else:
        # print(string, ' negative')
        return 'negative'


if __name__ == '__main__':

    #train()
    #string = '电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    # string = '牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    # string = '酒店的环境非常好，价格也便宜，值得推荐'
    # string = '手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    # string = '我是傻逼'
    # string = '屏幕较差，拍照也很粗糙。'
    # string = '质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    # string = '东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
    # string = '【日媒称中国或探讨解禁福岛食品 中国学者：符合标准才可能考虑】日本共同社1日独家爆料称，有“日中关系消息人士”透露，中国提议设工作组探讨解禁从日本进口福岛等地的食品。该报道认为，中方在进口涉福岛食品上的态度出现转变。中国社科院日本所学者卢昊1日对《环球时报》表示，目前官方没有公布相关消息，即使中方真的提议设立工作组，也只是第一步，是否解除进口限制，中方会对当地的农产品情况进行系统评估，符合中方标准的才可能考虑解禁。http://t.cn/RHWurwi[组图共2张]'
    # string = '你是傻逼'
    # string = '【中国“洋垃圾”进口禁令让英国无措：不能焚烧，又没能力处理】“英国马上就要面临塑料废物的堆积问题。”新年伊始，英国各大媒体集体发声。引发英国舆论的关键是中国对“洋垃圾”的进口禁令，其相关条例就在近日生效。据报道，去年7月国务院办公厅印发《禁止洋垃圾入境推进固体废物进口管理制度改革实施方案》，2017年年底前，禁止进口来自生活源的废塑料、未经分拣的废纸以及废纺织原料、钒渣等环境危害大、群众反映强烈的固体废物。2019年年底前，逐步停止进口国内资源可以替代的固体废物。目前，英国正在思考垃圾的未来去向，短期做法是将废物出口到越南和印度，但这些国家能够接受的规模有限，如果焚烧还将面临一系列环境问题。英国广播公司2日称，目前一切还没有定论，还不清楚英国将如何达成这种长期目标，以及如何解决中国禁令带来的短期危机。http://t.cn/RHY9fGd'
    #
    # string = '【中国不再进口塑料废品 英国人慌了】据BBC1月2日报道，英国过去每年都会运送50万吨塑料废料到中国，由后者作回收处理，但现在这种交易要终止了。中国在本月正式启动新法规，禁止进口所谓的“洋垃圾”，这是中国试图进行产业结构升级的一部分。而英国目前却不知道该如何应对。据报道，英国过去每年都会运送50万吨塑料废料到中国，由后者作回收处理，但现在这种交易要终止了。中国在本月正式启动新法规，禁止进口所谓的“洋垃圾”，这是中国试图进行产业结构升级的一部分。其他亚洲国家会接收一部分塑料，但仍将会有大量废料留下来。 via.香港商报网'
    # string = '【儿科神药匹多莫德遭扒皮】继莎普爱思滴眼液被曝光调查后，又一个“神药”被北京和睦家医院药师门诊主任冀连梅扒皮。据她透露，这个进口药在国外医学临床试验尚处于小白鼠阶段，疗效尚不明确，但在我国却摇身一变成了价格昂贵的“神药”↓↓她总结，该药存在四大问题↓↓儿科“神药”匹多莫德遭扒皮 国外尚处试验阶段'
    # string = '【中国全面禁洋垃圾！整个欧美一下崩溃成这样...】从2018年1月1日开始，中国禁止进口洋垃圾，这其中就包括废弃塑胶、纸类、废弃炉渣、与纺织品。中国在宣布对洋垃圾实施进口禁令后，西方措手不及，不知如何应对这一改变。因为，很多垃圾出口国没有充足的基础设施，难以充分实现对废旧物品及垃圾的回收利用。他们或者会去寻找其他亚洲国家作为垃圾堆放点，比如印度、孟加拉国或者越南。http://t.cn/RHEBaTy'

    content_comment = pickle.load(open('../../jinkou2.pkl', 'rb'))
    out = open('Senti_2.csv', 'a', newline='', encoding='gb18030')
    csv_write = csv.writer(out, dialect='excel')
    stu1 = ['微博创建时间', '微博url', '点赞数', '转发数', '评论数', '工具', '关键词', '微博内容', '情感正负']
    csv_write.writerow(stu1)

    count = 1
    for i in content_comment:
        print("处理到：", count)
        senti = []
        senti.append(i[0])
        senti.append(i[4])
        senti.append(i[5])
        senti.append(i[6])
        senti.append(i[7])
        senti.append(i[8])
        senti.append(i[3])
        senti.append(i[1])

        senti.append(lstm_predict(i[1]))
        print(senti)
        count += 1
        csv_write.writerow(senti)

    # train()
