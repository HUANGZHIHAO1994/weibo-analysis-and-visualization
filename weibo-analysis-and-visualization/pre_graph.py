'''
专门用于作关系图（graph.py）的数据预处理，以pre_graph2为准
用于匹配content、comment并初步清理数据
将每个分类关键词形式变为：
[
  [{node1}, {node2}, ... {noden}],
  [{link1}, {link2}, ... {linkm}],
  [{category1}, {category2}, ... {categoryx}]
]
并压缩成weibo.json文件

节点symbolSize和value数值的算法：
symbolSize为： 收到评论数去自然对数四舍五入取整(后乘以2)，若发了多条微博则求和，评论别人微博得0.2分
value为： 收到每条评论和评论别人都算1分

pyecharts graph的说明见：https://blog.csdn.net/Kevin_HZH/article/details/91043392
comment.json content.json 数据形式见 prepro.py文件
'''

import codecs
import json
import pandas as pd
import jieba
import pickle
import re
import math



def Match(comment, content):

    content_comment = []
    advertisement = ["王者荣耀", "券后", "售价", '¥', "￥", '下单']
    nodes = []
    links = []
    categories = []
    nodes1 = {}


    for k in range(0, len(content)):
        judge = []
        print('Processing train ', k)

        for adv in advertisement:
            if adv in content[k]['content']:
                judge.append("True")
                break
        if re.search(r"买.*赠.*", content[k]['content']):
            judge.append("True")
            continue
        #  以上是判断是否为广告微博
        if "True" not in judge:
            categories1 = {}
            categories1["name"] = content[k]['user_id']   #  name
            categories.append(categories1)

            try:
                symbolSize = int(round(math.log(int(content[k]['comment_num']['$numberInt']))) * 2)      #  symbolSize① 被别人评论
            except:
                symbolSize = 2
            value = int(content[k]['comment_num']['$numberInt'])

            if content[k]['user_id'] not in nodes1:
                nodes1[content[k]['user_id']] = {}
            nodes1[content[k]['user_id']]['symbolSize'] = nodes1[content[k]['user_id']].get('symbolSize', 0) + symbolSize   #  symbolSize① 被别人评论
            nodes1[content[k]['user_id']]['value'] = nodes1[content[k]['user_id']].get('value', 0) + value        #  value①  被别人评论
            url = content[k]['weibo_url']
            # nodes1["name"] = content[k]['user_id']
            nodes1[content[k]['user_id']]["draggable"] = "False"
            nodes1[content[k]['user_id']]["label"] = {
                "normal": {
                    "show": "True"
                }
            }

        #  print(content[k]['comment_num']['$numberInt'])

        #  nodes1["value"] = nodes1["symbolSize"]
            nodes1[content[k]['user_id']]["category"] = content[k]['user_id']     #  category


            for i in comment:
                if i['weibo_url'] == url:
                    if i["comment_user_id"] not in nodes1:
                        nodes1[i["comment_user_id"]] = {}

                    nodes1[i["comment_user_id"]]["symbolSize"] = nodes1[i["comment_user_id"]].get('symbolSize', 0) + 0.2   #  symbolSize② 评论别人
                    nodes1[i["comment_user_id"]]["value"] = nodes1[i["comment_user_id"]].get("value", 0) + 1      #   value② 评论别人

                    nodes1[i["comment_user_id"]]["draggable"] = "False"
                    nodes1[i["comment_user_id"]]["category"] = content[k]['user_id']

                    links1 = {}
                    links1["source"] = content[k]['user_id']
                    links1["target"] = i["comment_user_id"]
                    links.append(links1)
    for i in nodes1:
        nodes2 = {}
        nodes2["name"] = i
        if nodes1[i]['symbolSize'] < 2:
            nodes2["symbolSize"] = 2     #  symbolSize③  节点大小的下限
        else:
            nodes2["symbolSize"] = int(nodes1[i]['symbolSize'])
        nodes2["value"] = nodes1[i]['value']
        nodes2["draggable"] = "False"
        nodes2["category"] = nodes1[i]["category"]
        try:
            nodes2["label"] = nodes1[i]['label']
        except:
            pass

        nodes.append(nodes2)

    content_comment.append(nodes)
    content_comment.append(links)
    content_comment.append(categories)

    with open('weibo.json', 'w', encoding='utf-8') as f:
        json.dump(content_comment, f)
    # json.dump(content_comment, open('./jinkou.pkl', 'wb'))


if __name__ == '__main__':

    #  txt文档读法
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
    # pandas文档读法
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

    Match(comment, content)
