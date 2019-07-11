import pandas as pd
import matplotlib
import scipy
import matplotlib.pyplot as plt
from matplotlib.font_manager import *
from datetime import datetime
import matplotlib.dates as mdates

import pyecharts.options as opts
from example.commons import Collector, Faker
from pyecharts.charts import Line, Page

def date_deal(string):
    '''
    将时间显示从2018/1/1 23:59:59 转化为2018-01-01
    :param string: 如2018/1/1 23:59:59
    :return: string: 如2018/1/1
    '''
    string['微博创建时间'] = string['微博创建时间'].split(' ')[0]
    string['微博创建时间'] = str(datetime(year=int(string['微博创建时间'].split('/')[0]), month=int(string['微博创建时间'].split('/')[1]), day=int(string['微博创建时间'].split('/')[2])))
    string['微博创建时间'] = string['微博创建时间'].split(' ')[0]
    return string['微博创建时间']

df_weibo = pd.read_csv('Senti_Keyword_total_id.csv', sep=',', encoding='gb18030')
df_weibo = df_weibo.drop(['微博url', '工具', '微博内容', '情感得分', 'TF-IDF关键词', 'TextRank关键词', 'weibo_id'], axis=1)
df_weibo['微博创建时间'] = df_weibo.apply(date_deal, axis=1)
print(df_weibo.head())

# GroupBy = df_weibo.groupby(['关键词', '微博创建时间'])
# for i, j in GroupBy:
#     print(i)
#     print('*' * 40)
#     print(j)
# print(GroupBy.get_group('进口')['微博创建时间'])



GroupBy = df_weibo.groupby(['关键词', '微博创建时间']).sum()
# print(type(GroupBy))
# print(type(GroupBy.xs('进口')))
print(GroupBy.xs('进口'))
print(GroupBy.xs('进口').index[:])

#  从key_words2.txt文件中取出关键词用于匹配
f = open("./key_words2.txt", 'r', encoding='UTF-8-sig')
s = f.read()
s = s.replace('\n', '；')
s = s.replace(' ', '')
f.close()
#  print(s)
start_uids1 = s.split('；')[:-1]
start_uids = list(set(start_uids1))
start_uids.sort(key=start_uids1.index)

#  取出日期列
date1 = []
for i in GroupBy.xs('进口').index:
    date1.append(str(i))

# for uid in start_uids:
#     uid = uid.strip()
#     date2 = []
#     for i in GroupBy.xs(uid)['评论数']:
#         date2.append(int(i))

C = Collector()

@C.funcs
def line_base() -> Line:
    #   添加遍历的方法
    tmp = Line().add_xaxis(date1)
    for uid in start_uids:
        uid = uid.strip()
        try:
            date2 = []
            for i in GroupBy.xs(uid)['评论数']:
                date2.append(int(i))
            tmp.add_yaxis(uid, date2)
        except:
            continue

    #  调整title位置
    c = (
        tmp.set_global_opts(title_opts=opts.TitleOpts(title="评论数", pos_top='10%', pos_left="center"),
                            legend_opts=opts.LegendOpts(type_='scroll')
                            )
        # legend_opts=opts.LegendOpts(orient="orient")
    )

    return c

@C.funcs
def line_base1() -> Line:
    tmp = Line().add_xaxis(date1)
    for uid in start_uids:
        uid = uid.strip()
        try:
            date2 = []
            for i in GroupBy.xs(uid)['点赞数']:
                date2.append(int(i))
            tmp.add_yaxis(uid, date2)
        except:
            continue

    c = (
        tmp.set_global_opts(title_opts=opts.TitleOpts(title="点赞数", pos_top='10%', pos_left="center"),
                            legend_opts=opts.LegendOpts(type_='scroll')
                            )
        # legend_opts=opts.LegendOpts(orient="orient")
    )

    return c

@C.funcs
def line_base2() -> Line:
    tmp = Line().add_xaxis(date1)
    for uid in start_uids:
        uid = uid.strip()
        try:
            date2 = []
            for i in GroupBy.xs(uid)['转发数']:
                date2.append(int(i))
            tmp.add_yaxis(uid, date2)
        except:
            continue

    c = (
        tmp.set_global_opts(title_opts=opts.TitleOpts(title="转发数", pos_top='10%', pos_left="center"),
                            legend_opts=opts.LegendOpts(type_='scroll')
                            )
        # legend_opts=opts.LegendOpts(orient="orient")
    )

    return c


Page().add(*[fn() for fn, _ in C.charts]).render(u'./line.html')

