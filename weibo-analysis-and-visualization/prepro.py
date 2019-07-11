# -*- coding: utf-8 -*
'''
匹配content、comment并初步清理数据
用于作词云wc.py、地图graph.py、Tf-idf文本聚类cluster_tfidf、Word2Vec文本聚类cluster_w2v
将每个分类关键词形式变为：
[
  [url1, content1原文, [content1分词],[comment1],...,[comment_n]],
  [url2, content2原文, [content2分词],[comment1],...,[comment_m]],
  ...
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


def Match(comment, content):
    """匹配微博内容和微博评论数据，并将明显的广告微博剔除

    Args:
        comment_example:
      [
      {'_id': 'C_4322161898716112', 'crawl_time': '2019-06-01 20:35:36', 'weibo_url': 'https://weibo.com/1896820725/H9inNf22b', 'comment_user_id': '6044625121', 'content': '没问题，', 'like_num': {'$numberInt': '0'}, 'created_at': '2018-12-28 11:19:21'},...
      ]

        content_example:
      [
      {'_id': '1177737142_H4PSVeZWD', 'keyword': 'A股', 'crawl_time': '2019-06-01 20:31:13', 'weibo_url': 'https://weibo.com/1177737142/H4PSVeZWD', 'user_id': '1177737142', 'created_at': '2018-11-29 03:02:30', 'tool': 'Android', 'like_num': {'$numberInt': '0'}, 'repost_num': {'$numberInt': '0'}, 'comment_num': {'$numberInt': '0'}, 'image_url': 'http://wx4.sinaimg.cn/wap180/4632d7b6ly1fxod61wktyj20u00m8ahf.jpg', 'content': '#a股观点# 鲍威尔主席或是因为被特朗普总统点名批评后萌生悔改之意，今晚一番讲话被市场解读为美联储或暂停加息步伐。美元指数应声下挫，美股及金属贵金属价格大幅上扬，A50表现也并不逊色太多。对明天A股或有积极影响，反弹或能得以延续。 [组图共2张]'},...
      ]

    Returns:
        其实没有return，形成如下格式的pkl文件：
        [
        [url1, content1原文, [content1分词],[comment1],...,[comment_n]],
        [url2, content2原文, [content2分词],[comment1],...,[comment_m]],
        ...
        ]
    """
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
        # 通过上面的两种模式判断是不是广告
        if "True" not in judge:
            comment_list = []
            url = content[k]['weibo_url']
            comment_list.append(url)
            comment_list.append(content[k]['content'])
            # 数据清洗
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
            for i in comment:
                if i['weibo_url'] == url:    #  通过URL匹配content和comment
                    a1 = re.compile(r'回复@.*:')
                    i['content'] = a1.sub('', i['content'])
                    i['content'] = Traditional2Simplified(i['content'])
                    i['content'] = a2.sub('', i['content'])
                    i['content'] = a4.sub('', i['content'])
                    i['content'] = a5.sub('', i['content'])
                    i['content'] = a6.sub('', i['content'])
                    comment_list.append(Sent2Word(i['content']))
            content_comment.append(comment_list)

    pickle.dump(content_comment, open('./Agu.pkl', 'wb'))


if __name__ == '__main__':

    print("停用词读取")
    stop_words = [w.strip() for w in open('./dict/哈工大停用词表.txt', 'r', encoding='UTF-8').readlines()]
    stop_words.extend(['\n', '\t', ' ', '回复', '转发微博', '转发', '微博', '秒拍', '秒拍视频', '视频', "王者荣耀", "王者", "荣耀"])
    for i in range(128000, 128722 + 1):
        stop_words.extend(chr(i))
    stop_words.extend(['A股'])

    # 特殊字符

    # print(ord("🐏"))
    # print(chr(77823))
    # print(hex(128300))
    # print(0x1E000)
    # # E000-F8FF 122880-129279  # 自行使用區域 (Private Use Zone)
    # # F900-FAFF 129280-129791
    # count = 0
    # for i in range(128000, 129279+1):
    #     count += 1
    #     if count <100:
    #         print(chr(i))
    #     else:
    #         break

    #  json文档读法
    print("comment读取")
    f = codecs.open('./Agu_comment.json', 'r', 'UTF-8-sig')
    comment = []
    for i in f.readlines():
        try:
            comment.append(eval(i))
        except:
            continue
    # comment = [json.loads(i) for i in f.readlines()]  # json.loads也行
    f.close()
    # print(comment)

    '''
      comment_example:
      [
      {'_id': 'C_4322161898716112', 'crawl_time': '2019-06-01 20:35:36', 'weibo_url': 'https://weibo.com/1896820725/H9inNf22b', 'comment_user_id': '6044625121', 'content': '没问题，', 'like_num': {'$numberInt': '0'}, 'created_at': '2018-12-28 11:19:21'},...
      ]
    '''

    # pandas csv文档读法
    # df_weibo = pd.read_csv('comment.csv', sep=',', quotechar='"', error_bad_lines=False)
    # print(df_weibo.head())

    print("content读取")
    f = codecs.open('./Agu_content.json', 'r', 'UTF-8-sig')
    content = []
    for i in f.readlines():
        try:
            content.append(eval(i))
        except:
            continue
    # jinkou = [json.loads(i) for i in f.readlines()]  # json.loads也行
    f.close()
    # print(content)

    '''
      content_example:
      [
      {'_id': '1177737142_H4PSVeZWD', 'keyword': 'A股', 'crawl_time': '2019-06-01 20:31:13', 'weibo_url': 'https://weibo.com/1177737142/H4PSVeZWD', 'user_id': '1177737142', 'created_at': '2018-11-29 03:02:30', 'tool': 'Android', 'like_num': {'$numberInt': '0'}, 'repost_num': {'$numberInt': '0'}, 'comment_num': {'$numberInt': '0'}, 'image_url': 'http://wx4.sinaimg.cn/wap180/4632d7b6ly1fxod61wktyj20u00m8ahf.jpg', 'content': '#a股观点# 鲍威尔主席或是因为被特朗普总统点名批评后萌生悔改之意，今晚一番讲话被市场解读为美联储或暂停加息步伐。美元指数应声下挫，美股及金属贵金属价格大幅上扬，A50表现也并不逊色太多。对明天A股或有积极影响，反弹或能得以延续。 [组图共2张]'},...
      ]
    '''

    Match(comment, content)


