# -*- coding: utf-8 -*-

import yaml
import sys
from smart_open import open
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
from keras.layers.core import Dense, Dropout,Activation
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import jieba
import pandas as pd
import sys
sys.setrecursionlimit(1000000)
# set parameters:
vocab_dim = 100
maxlen = 100
n_iterations = 1  # ideally more..
n_exposures = 5
window_size = 7
batch_size = 32
n_epoch = 4
input_length = 100
cpu_count = multiprocessing.cpu_count()


#加载训练文件
def loadfile():
    neg=pd.read_excel('../data/neg.xls',header=None,index=None)
    pos=pd.read_excel('../data/pos.xls',header=None,index=None)

    combined=np.concatenate((pos[0], neg[0]))
    # print(combined)
    # print(combined.shape)
    y = np.concatenate((np.ones(len(pos),dtype=int), np.zeros(len(neg),dtype=int)))

    return combined, y

#对句子经行分词，并去掉换行符
def tokenizer(text):
    '''

    :param text:  string
    :return:  [[tokenize1,tokenize2..]] list of list
    '''
    text = [jieba.lcut(document.replace('\n', '')) for document in text]
    return text



#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def create_dictionaries(model=None,
                        combined=None):
    ''' Function does are number of Jobs:
        1- Creates a word to index mapping
        2- Creates a word to vector mapping
        3- Transforms the Training and Testing Dictionaries

    '''
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab,
                            allow_update=True)
        w2indx = {v: k+1 for k, v in gensim_dict.items()} # 所有频数超过10的词语的索引
        w2vec = {word: model[word] for word in w2indx.keys()} # 所有频数超过10的词语的词向量

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
        combined = sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


#创建词语字典，并返回每个词语的索引，词向量，以及每个句子所对应的词语索引
def word2vec_train(combined):

    model = Word2Vec(size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     iter=n_iterations)
    model.build_vocab(combined)
    model.train(combined, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('../lstm_data/W2Vmodel')
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined

def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  # 所有单词的索引数，频数小于10的词语索引为0，所以加1
    embedding_weights = np.zeros((n_symbols, vocab_dim))#索引为0的词语，词向量全为0
    for word, index in index_dict.items(): #从索引为1的词语开始，对每个词语对应其词向量
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.2)
    print(x_train.shape,y_train.shape)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test


##定义网络结构
def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
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
    model.fit(x_train, y_train, batch_size=batch_size, epochs=n_epoch, verbose=1, validation_data=(x_test, y_test))

    print("Evaluate...")
    score = model.evaluate(x_test, y_test, batch_size=batch_size)

    yaml_string = model.to_yaml()
    with open('../lstm_data/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('../lstm_data/lstm.h5')
    print('Test score:', score)


#训练模型，并保存
def train():
    print('Loading Data...')
    combined, y = loadfile()
    print(len(combined),len(y))
    print('Tokenising...')
    combined = tokenizer(combined)
    print('Training a Word2vec model...')
    index_dict, word_vectors, combined = word2vec_train(combined)
    print('Setting up Arrays for Keras Embedding Layer...')
    n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
    print(x_train.shape,y_train.shape)
    train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)




def input_transform(string):
    words=jieba.lcut(string)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('../lstm_data/W2Vmodel')
    # 举例子，相近词

    for key in model.similar_by_word(u'日本', topn=10):
        # if len(key[0]) == 3:  # key[0]应该就表示某个词
        print(key[0], key[1])  # 某一个词,某一个词出现的概率

    _,_,combined=create_dictionaries(model,words)
    return combined

def lstm_predict(string):
    print('loading model......')
    with open('../lstm_data/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print( 'loading weights......')
    model.load_weights('../lstm_data/lstm.h5')
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print('Word2Vec......')
    data = input_transform(string)
    data.reshape(1, -1)
    print(data)
    print(len(data[0]))
    result = model.predict_classes(data)
    if result[0][0] == 1:
        print(string, ' positive')
    else:
        print(string, ' negative')


if __name__ == '__main__':

    #train()
    #string='电池充完了电连手机都打不开.简直烂的要命.真是金玉其外,败絮其中!连5号电池都不如'
    string='牛逼的手机，从3米高的地方摔下去都没坏，质量非常好'
    string='酒店的环境非常好，价格也便宜，值得推荐'
    string='手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了'
    string='我是傻逼'
    string='你是傻逼'
    string='屏幕较差，拍照也很粗糙。'
    string='质量不错，是正品 ，安装师傅也很好，才要了83元材料费'
    string='东西非常不错，安装师傅很负责人，装的也很漂亮，精致，谢谢安装师傅！'
    string = '【日媒称中国或探讨解禁福岛食品 中国学者：符合标准才可能考虑】日本共同社1日独家爆料称，有“日中关系消息人士”透露，中国提议设工作组探讨解禁从日本进口福岛等地的食品。该报道认为，中方在进口涉福岛食品上的态度出现转变。中国社科院日本所学者卢昊1日对《环球时报》表示，目前官方没有公布相关消息，即使中方真的提议设立工作组，也只是第一步，是否解除进口限制，中方会对当地的农产品情况进行系统评估，符合中方标准的才可能考虑解禁。http://t.cn/RHWurwi[组图共2张]'

    lstm_predict(string)

    # train()
