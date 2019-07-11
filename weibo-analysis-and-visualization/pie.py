import pandas as pd
from numpy import max, min, mean, quantile, median, percentile

from example.commons import Collector, Faker
from pyecharts import options as opts
from pyecharts.charts import Page, Pie

df_weibo = pd.read_csv('Senti_Keyword_total_id.csv', sep=',', encoding='gb18030')
df_weibo = df_weibo.drop(['微博url', '工具', '微博内容', '点赞数', '转发数', '评论数', '微博创建时间', 'TF-IDF关键词', 'TextRank关键词', 'weibo_id'], axis=1)

#  winsorize 去除极端值
q5 = df_weibo.quantile(0.05)['情感得分']
q95 = df_weibo.quantile(0.95)['情感得分']
# print(df_weibo.quantile(0.95)['情感得分'])

senti_list = []
for i in df_weibo['情感得分']:
    if (i <= q95) and (i >= q5):
        senti_list.append(i)

senti_dict = {}
# 分位数统计，统计结果有用于人工将情感得分划分出区间
senti_max = max(senti_list)
senti_min = min(senti_list)
senti_mean = mean(senti_list)
senti_25 = (senti_mean - senti_min) / 2
senti_75 = (senti_max - senti_mean) / 2
print(senti_max, senti_min, senti_mean, senti_25, senti_75)

for i in senti_list:
    if i < -6:
        senti_dict['极端负面情感'] = senti_dict.get("极端负面情感", 0) + 1
    elif -6 <= i < -2:
        senti_dict['负面情感'] = senti_dict.get("负面情感", 0) + 1
    elif -2 <= i < 5:
        senti_dict['中性情感'] = senti_dict.get("中性情感", 0) + 1
    elif 5 <= i < 25:
        senti_dict['正向情感'] = senti_dict.get("正向情感", 0) + 1
    elif 25 <= i:
        senti_dict['极端正向情感'] = senti_dict.get("极端正向情感", 0) + 1

data = []
for k, v in senti_dict.items():
    data1 = []
    data1.append(k)
    data1.append(v)
    data.append(data1)

C = Collector()


@C.funcs
def pie_base() -> Pie:
    c = (
        Pie()
        .add("", data)
        .set_global_opts(title_opts=opts.TitleOpts(title="Pie-情感分析"))
        .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
    )
    return c


# @C.funcs
# def pie_set_colors() -> Pie:
#     c = (
#         Pie()
#         .add("", [list(z) for z in zip(Faker.choose(), Faker.values())])
#         .set_colors(["blue", "green", "yellow", "red", "pink", "orange", "purple"])
#         .set_global_opts(title_opts=opts.TitleOpts(title="Pie-设置颜色"))
#         .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
#     )
#     return c
#
#
# @C.funcs
# def pie_position() -> Pie:
#     c = (
#         Pie()
#         .add(
#             "",
#             [list(z) for z in zip(Faker.choose(), Faker.values())],
#             center=["35%", "50%"],
#         )
#         .set_global_opts(
#             title_opts=opts.TitleOpts(title="Pie-调整位置"),
#             legend_opts=opts.LegendOpts(pos_left="15%"),
#         )
#         .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
#     )
#     return c
#
#
# @C.funcs
# def pie_radius() -> Pie:
#     c = (
#         Pie()
#         .add(
#             "",
#             [list(z) for z in zip(Faker.choose(), Faker.values())],
#             radius=["40%", "75%"],
#         )
#         .set_global_opts(
#             title_opts=opts.TitleOpts(title="Pie-Radius"),
#             legend_opts=opts.LegendOpts(
#                 orient="vertical", pos_top="15%", pos_left="2%"
#             ),
#         )
#         .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
#     )
#     return c
#
#
# @C.funcs
# def pie_rosetype() -> Pie:
#     v = Faker.choose()
#     c = (
#         Pie()
#         .add(
#             "",
#             [list(z) for z in zip(v, Faker.values())],
#             radius=["30%", "75%"],
#             center=["25%", "50%"],
#             rosetype="radius",
#             label_opts=opts.LabelOpts(is_show=False),
#         )
#         .add(
#             "",
#             [list(z) for z in zip(v, Faker.values())],
#             radius=["30%", "75%"],
#             center=["75%", "50%"],
#             rosetype="area",
#         )
#         .set_global_opts(title_opts=opts.TitleOpts(title="Pie-玫瑰图示例"))
#     )
#     return c
#
#
# @C.funcs
# def pie_scroll_legend() -> Pie:
#     c = (
#         Pie()
#         .add(
#             "",
#             [
#                 list(z)
#                 for z in zip(
#                     Faker.choose() + Faker.choose() + Faker.choose(),
#                     Faker.values() + Faker.values() + Faker.values(),
#                 )
#             ],
#             center=["40%", "50%"],
#         )
#         .set_global_opts(
#             title_opts=opts.TitleOpts(title="Pie-Legend 滚动"),
#             legend_opts=opts.LegendOpts(
#                 type_="scroll", pos_left="80%", orient="vertical"
#             ),
#         )
#         .set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}"))
#     )
#     return c
#
#
# @C.funcs
# def pie_rich_label() -> Pie:
#     c = (
#         Pie()
#         .add(
#             "",
#             [list(z) for z in zip(Faker.choose(), Faker.values())],
#             radius=["40%", "55%"],
#             label_opts=opts.LabelOpts(
#                 position="outside",
#                 formatter="{a|{a}}{abg|}\n{hr|}\n {b|{b}: }{c}  {per|{d}%}  ",
#                 background_color="#eee",
#                 border_color="#aaa",
#                 border_width=1,
#                 border_radius=4,
#                 rich={
#                     "a": {"color": "#999", "lineHeight": 22, "align": "center"},
#                     "abg": {
#                         "backgroundColor": "#e3e3e3",
#                         "width": "100%",
#                         "align": "right",
#                         "height": 22,
#                         "borderRadius": [4, 4, 0, 0],
#                     },
#                     "hr": {
#                         "borderColor": "#aaa",
#                         "width": "100%",
#                         "borderWidth": 0.5,
#                         "height": 0,
#                     },
#                     "b": {"fontSize": 16, "lineHeight": 33},
#                     "per": {
#                         "color": "#eee",
#                         "backgroundColor": "#334455",
#                         "padding": [2, 4],
#                         "borderRadius": 2,
#                     },
#                 },
#             ),
#         )
#         .set_global_opts(title_opts=opts.TitleOpts(title="Pie-富文本示例"))
#     )
#     return c


Page().add(*[fn() for fn, _ in C.charts]).render(u'./pie.html')
# print([list(z) for z in zip(Faker.choose(), Faker.values())])
