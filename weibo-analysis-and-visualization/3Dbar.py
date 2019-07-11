import random
from datetime import datetime
import pandas as pd

from example.commons import Collector, Faker
from pyecharts import options as opts
from pyecharts.charts import Bar3D, Page

'''
生成2018年日期和365数字对应dict形式：
{'2018-01-01':1, '2018-01-02':2,}
'''
date1 = []
for i in pd.date_range('01/01/2018', '12/31/2018'):
    # date1.append(str(i).replace("-", '').split()[0])
    date1.append(str(i).split()[0])
print(len(date1))

date_dict = {}
for index, date in enumerate(date1):
    # index += 1
    date_dict[date] = index

'''
同样生成关键词的对应dict
{'进口':1, '出口':2, ...}
'''
f = open("./key_words2.txt", 'r', encoding='UTF-8-sig')
s = f.read()
s = s.replace('\n', '；')
s = s.replace(' ', '')
f.close()
#         # print(s)
start_uids1 = s.split('；')[:-1]
start_uids = list(set(start_uids1))
start_uids.sort(key=start_uids1.index)
key_dict = {}
for index, date in enumerate(start_uids):
    # index += 1
    key_dict[date] = index

def date_deal(string):
    string['微博创建时间'] = string['微博创建时间'].split(' ')[0]

    string['微博创建时间'] = str(datetime(year=int(string['微博创建时间'].split('/')[0]), month=int(string['微博创建时间'].split('/')[1]), day=int(string['微博创建时间'].split('/')[2])))
    string['微博创建时间'] = string['微博创建时间'].split(' ')[0]
    string['微博创建时间'] = date_dict[string['微博创建时间']]
    return string['微博创建时间']



def key_deal(string):

    string['关键词'] = key_dict[string['关键词'].strip()]

    return string['关键词']

def senti_deal(string):

    string['情感得分'] = int(string['情感得分'])

    return string['情感得分']


df_weibo = pd.read_csv('Senti_Keyword_total_id.csv', sep=',', encoding='gb18030')
df_weibo = df_weibo.drop(['微博url', '工具', '微博内容', '点赞数', '转发数', '评论数', 'TF-IDF关键词', 'TextRank关键词', 'weibo_id'], axis=1)
df_weibo['微博创建时间'] = df_weibo.apply(date_deal, axis=1)
df_weibo['关键词'] = df_weibo.apply(key_deal, axis=1)
df_weibo['情感得分'] = df_weibo.apply(senti_deal, axis=1)

print(df_weibo.describe())

GroupBy = df_weibo.groupby(['关键词', '微博创建时间']).mean()
print(GroupBy.xs(1))

data = []
for uid in range(82):
    dat = []
    for d in GroupBy.xs(uid).index:
        dat.append(d)
    # 3Dbar要求365天都有，因此当天没有微博内容的赋值为零
    for j in range(365):
        date2 = []
        if j in dat:
            date2.append(j)
            date2.append(uid)
            #  为了可视化效果，将情感得分>1000的截断
            if int(GroupBy.xs(uid)['情感得分'][j]) > 1000:
                date2.append(1000)
            else:
                date2.append(int(GroupBy.xs(uid)['情感得分'][j]))
            data.append(date2)
        else:
            date2.append(j)
            date2.append(uid)
            date2.append(0)
            data.append(date2)

print(data)

C = Collector()


@C.funcs
def bar3d_base() -> Bar3D:
    # data = [(i, j, random.randint(0, 12)) for i in range(6) for j in range(24)]
    # print([[d[1], d[0], d[2]] for d in data])
    c = (
        Bar3D(init_opts=opts.InitOpts(width='1200px', height='1000px'))
        .add(
            "",
            data,
            xaxis3d_opts=opts.Axis3DOpts(date1, type_="category"),
            yaxis3d_opts=opts.Axis3DOpts(start_uids, type_="category"),
            zaxis3d_opts=opts.Axis3DOpts(type_="value"),
        )
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(max_=500),
            title_opts=opts.TitleOpts(title="情感分析3D"),
        )
    )
    return c


Page().add(*[fn() for fn, _ in C.charts]).render(u'./3Dbar.html')
# 打印最大值对应关键词和最小值对应关键词
print(start_uids[33])
print(start_uids[75])

