'''
这个程序文件是总括类型的，包括了基于词典的情感分析、调包snownlp的情感分析、Textrank关键词提取、Tfidf关键词提取
根据需要使用
注意：snownlp的速度会很慢
'''


import numpy as np
from collections import defaultdict
import pickle
import heapq
from jieba import analyse
import csv
from snownlp import SnowNLP

def LoadDict():
    """Load Dict form disk

    Returns:
        senti_word: senti word dict (BosonNLP)
        not_word: not word dict  （情感极性词典.zip）
        degree_word: degree word dict （大连理工）
    """
    # Sentiment word
    senti_file = open('./dict/BosonNLP_sentiment_score.txt', 'r+', encoding='utf-8')
    # 获取字典内容
    # 去除'\n'
    senti_list = senti_file.read().splitlines()
    # 创建情感字典
    senti_dict = defaultdict()
    # 读取字典文件每一行内容，将其转换为字典对象，key为情感词，value为对应的分值
    for s in senti_list:
        # 对每一行内容根据空格分隔，索引0是情感词，1是情感分值
        if len(s.split(' ')) == 2:
            senti_dict[s.split(' ')[0]] = s.split(' ')[1]

    not_words = open('./dict/否定词.txt', encoding='UTF-8').readlines()
    not_dict = {}
    for w in not_words:
        word = w.strip()
        not_dict[word] = float(-1)

    degree_words = open('./dict/degreeDict.txt', 'r+', encoding='utf-8').readlines()
    degree_dict = {}
    for w in degree_words:
        word, score = w.strip().split(',')
        degree_dict[word] = float(score)

    return senti_dict, not_dict, degree_dict


senti_dict, not_dict, degree_dict = LoadDict()


def LocateSpecialWord(senti_dict, not_dict, degree_dict, sent):
    """Find the location of Sentiment words, Not words, and Degree words

    The idea is pretty much naive, iterate every word to find the location of Sentiment words,
    Not words, and Degree words, additionally, storing the index into corresponding arrays
    SentiLoc, NotLoc, DegreeLoc.

    Args:
        senti_word: BosonNLP
        not_word: not word location dict  （情感极性词典.zip）
        degree_word: degree word location dict （大连理工）

    Returns:
        senti_word: senti word location dict, with word and their location in the sentence
        not_word: not word location dict
        degree_word: degree word location dict
    """
    senti_word = {}
    not_word = {}
    degree_word = {}

    for index, word in enumerate(sent):
        if word in senti_dict:
            senti_word[index] = senti_dict[word]
        elif word in not_dict:
            not_word[index] = -1.0
        elif word in degree_dict:
            degree_word[index] = degree_dict[word]

    return senti_word, not_word, degree_word


def ScoreSent(senti_word, not_word, degree_word, words):
    """Compute the sentiment score of this sentence
        基于词典的情感分析算法

    Args:
        senti_word: BosonNLP
        not_word: not word location dict  （情感极性词典.zip）
        degree_word: degree word location dict （大连理工）
        words: The tokenized word list.

    Returns:
        score: The sentiment score
    """
    W = 1
    score = 0

    # The location of sentiment words
    senti_locs = list(senti_word.keys())
    not_locs = list(not_word.keys())
    degree_locs = list(degree_word.keys())

    sentiloc = -1

    # iterate every word, i is the word index ("location")
    for i in range(0, len(words)):
        # if the word is positive
        if i in senti_locs:
            sentiloc += 1
            # update sentiment score
            score += W * float(senti_word[i])

            if sentiloc < len(senti_locs) - 1:
                # if there exists Not words or Degree words between
                # this sentiment word and next sentiment word
                # j is the word index ("location")
                for j in range(senti_locs[sentiloc], senti_locs[sentiloc + 1]):
                    # if there exists Not words
                    if j in not_locs:
                        W *= -1
                    # if there exists degree words
                    elif j in degree_locs:
                        W *= degree_word[j]

    return score

if __name__ == '__main__':
    content_comment = pickle.load(open('./jinkou.pkl', 'rb'))
    print('='*40)
    print('1. 自编版情感分析')
    print('-'*40)
    print('Sentiment Analysis')
    print('-'*40)
    senti_dict, not_dict, degree_dict = LoadDict()
    senti = []
    for i in content_comment:
        if len(i[2]) > 4:
            senti_word, not_word, degree_word = LocateSpecialWord(senti_dict, not_dict, degree_dict, i[2])
            score = ScoreSent(senti_word, not_word, degree_word, i[2])
        else:
            score = 0
        senti.append(score)
    index1 = list(map(senti.index, heapq.nlargest(30, senti)))
    max_score = heapq.nlargest(30, senti)
    print(max_score, index1)
    for i in index1:
        print(content_comment[i])
    index2 = list(map(senti.index, heapq.nsmallest(30, senti)))
    min_score = heapq.nsmallest(30, senti)
    print(min_score, index2)
    for i in index2:
        print(content_comment[i])


    print('=' * 40)
    print('2. snownlp版')
    print('-' * 40)
    print('Sentiment Analysis')
    print('-' * 40)
    senti2 = []
    for i in content_comment:
        if len(i[2]) > 4:
            # i[1] = i[1].encode("utf-8")
            s = SnowNLP(i[1])
            score = s.sentiments
        else:
            score = 0
        senti2.append(score)
    index3 = list(map(senti2.index, heapq.nlargest(30, senti2)))
    max_score = heapq.nlargest(30, senti2)
    print(max_score, index3)
    for i in index3:
        print(content_comment[i])
    index4 = list(map(senti2.index, heapq.nsmallest(30, senti2)))
    min_score = heapq.nsmallest(30, senti2)
    print(min_score, index4)
    for i in index4:
        print(content_comment[i])

    '''
    关键词提取
    '''

    print('='*40)
    print('3. 关键词提取')
    print('-'*40)
    print(' TF-IDF')
    print('-'*40)

    tf_idf = []
    for i in content_comment:
        word_tf_idf = []
        for x, w in analyse.extract_tags(i[1], topK=5, withWeight=True):
            word_tf_idf.append(x)
        tf_idf.append(word_tf_idf)
    print(tf_idf)

    print('-'*40)
    print(' TextRank')
    print('-'*40)

    textrank = []
    for i in content_comment:
        word_textrank = []
        for x, w in analyse.textrank(i[1], topK=5, withWeight=True):
            word_textrank.append(x)
        textrank.append(word_textrank)
    print(textrank)
