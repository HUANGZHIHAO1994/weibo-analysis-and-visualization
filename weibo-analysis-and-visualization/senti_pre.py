'''
用于情感分析初步清理数据形成content2.pkl
所需的content.json是全关键词的微博内容
content2.pkl形式为：
[
  [created_at, content1原文, [content1分词], keyword, weibo_url, like_num, repost_num, comment_num, tool]
]
'''

import codecs
import json
import pandas as pd
import jieba
import pickle
import re
from langconv import *

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence


def Sent2Word(sentence):
    """Turn a sentence into tokenized word list and remove stop-word

    Using jieba to tokenize Chinese.

    Args:
        sentence: A string.

    Returns:
        words: A tokenized word list.
    """
    global stop_words

    words = jieba.cut(sentence)
    words = [w for w in words if w not in stop_words]

    return words


def Prepro(content):

    content_comment = []
    advertisement = ["王者荣耀", "券后", "售价", '¥', "￥", '下单']

    for k in range(0, len(content)):
        judge = []
        print('Processing train ', k)
        content[k]['content'] = Traditional2Simplified(content[k]['content'])
        for adv in advertisement:
            if adv in content[k]['content']:
                judge.append("True")
                break
        if re.search(r"买.*赠.*", content[k]['content']):
            judge.append("True")
            continue
        '''         
            [
              [created_at, content1原文, [content1分词], keyword, weibo_url, like_num, repost_num, comment_num, tool]  
            ]
        '''
        if "True" not in judge:
            comment_list = []
            comment_list.append(content[k]['created_at'])
            comment_list.append(content[k]['content'])
            a2 = re.compile(r'#.*?#')
            content[k]['content'] = a2.sub('', content[k]['content'])
            a3 = re.compile(r'\[组图共.*张\]')
            content[k]['content'] = a3.sub('', content[k]['content'])
            a4 = re.compile(r'http:.*')
            content[k]['content'] = a4.sub('', content[k]['content'])
            a5 = re.compile(r'@.*? ')
            content[k]['content'] = a5.sub('', content[k]['content'])
            a6 = re.compile(r'\[.*?\]')
            content[k]['content'] = a6.sub('', content[k]['content'])
            comment_list.append(Sent2Word(content[k]['content']))
            comment_list.append(content[k]['keyword'])
            url = content[k]['weibo_url']
            comment_list.append(url)
            try:
                comment_list.append(content[k]['like_num']["$numberInt"])
            except:
                comment_list.append(' ')
            try:
                comment_list.append(content[k]['repost_num']["$numberInt"])
            except:
                comment_list.append(' ')
            try:
                comment_list.append(content[k]['comment_num']["$numberInt"])
            except:
                comment_list.append(' ')

            try:
                comment_list.append(content[k]['tool'])
            except:
                comment_list.append(' ')
            content_comment.append(comment_list)

    pickle.dump(content_comment, open('./content2.pkl', 'wb'))


if __name__ == '__main__':

    print("停用词读取")
    stop_words = [w.strip() for w in open('./dict/哈工大停用词表.txt', 'r', encoding='UTF-8').readlines()]
    stop_words.extend(['\n', '\t', ' ', '回复', '转发微博', '转发', '微博', '秒拍', '秒拍视频', '视频', "王者荣耀", "王者", "荣耀"])
    for i in range(128000, 128722 + 1):
        stop_words.extend(chr(i))
#     stop_words.extend(['进口'])


    print("content读取")
    f = codecs.open('./content.json', 'r', 'UTF-8-sig')
    content = []
    for i in f.readlines():
        try:
            content.append(eval(i))
        except:
            continue
    # jinkou = [json.loads(i) for i in f.readlines()]  # json.loads也行
    f.close()

    Prepro(content)

