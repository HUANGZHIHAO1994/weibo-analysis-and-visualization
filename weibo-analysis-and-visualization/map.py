from example.commons import Collector, Faker
from pyecharts import options as opts
from pyecharts.charts import Geo, Page
from pyecharts.globals import ChartType, SymbolType

import numpy as np
import pickle
content_comment = pickle.load(open('./jinkou.pkl', 'rb'))

provin = '北京市，天津市，上海市，重庆市，河北省，山西省，辽宁省，吉林省，黑龙江省，江苏省，浙江省，安徽省，福建省，江西省，山东省，河南省，湖北省，湖南省，广东省，海南省，四川省，贵州省，云南省，陕西省，甘肃省，青海省，台湾省，内蒙古自治区，广西壮族自治区，西藏自治区，宁夏回族自治区，新疆维吾尔自治区，香港特别行政区，澳门特别行政区，北京，天津，上海，重庆，河北，山西，辽宁，吉林，黑龙江，江苏，浙江，安徽，福建，江西，山东，河南，湖北，湖南，广东，海南，四川，贵州，云南，陕西，甘肃，青海，台湾，内蒙古，广西，西藏，宁夏，新疆，香港，澳门'
provin = provin.split('，')
count = {}

for i in content_comment:
    for word in i[2]:
        if word in provin:
            count[word] = count.get(word, 0) + 1

count2 = {}
provin1 = '北京，天津，上海，重庆，河北，山西，辽宁，吉林，黑龙江，江苏，浙江，安徽，福建，江西，山东，河南，湖北，湖南，广东，海南，四川，贵州，云南，陕西，甘肃，青海，台湾，内蒙古，广西，西藏，宁夏，新疆，香港，澳门'
provin2 = '北京市，天津市，上海市，重庆市，河北省，山西省，辽宁省，吉林省，黑龙江省，江苏省，浙江省，安徽省，福建省，江西省，山东省，河南省，湖北省，湖南省，广东省，海南省，四川省，贵州省，云南省，陕西省，甘肃省，青海省，台湾省，内蒙古自治区，广西壮族自治区，西藏自治区，宁夏回族自治区，新疆维吾尔自治区，香港特别行政区，澳门特别行政区'
provin1 = provin1.split('，')
provin2 = provin2.split('，')

# 将上海和上海市，北京和北京市等进行汇总
for i in zip(provin1, provin2):
    count2[i[0]] = count.get(i[0], 0) + count.get(i[1], 0)


values = []
for i in count2.values():
    values.append(i)

# 将所有城市的热度值做归一化处理，按照热度最大城市为100进行同比例调整
values = np.array(values)

try:
    Max = values.max()

    Min = values.min()
    if Max != Min:
        values = ((values - Min) / (Max - Min) * 100).astype(np.int)
    else:
        pass
except:
    pass
values = values.tolist()


provin_list = []
for key, value in zip(provin1, values):
    list1 = [key, value]
    provin_list.append(list1)

print(provin_list)

# provin_list = []
# for key, value in count2.items():
#     list1 = [key, value]
#     provin_list.append(list1)

##############################################
# 上面是数据预处理，下面是作图


C = Collector()


@C.funcs
def geo_base() -> Geo:
    c = (
        Geo()
        .add_schema(maptype="china")
        .add("城市", provin_list)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(),
            title_opts=opts.TitleOpts(title="微博城市热度分析"),
        )
    )
    return c


@C.funcs
def geo_visualmap_piecewise() -> Geo:
    c = (
        Geo()
        .add_schema(maptype="china")
        .add("城市", provin_list)
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(is_piecewise=True),
            title_opts=opts.TitleOpts(title="微博城市热度分析"),
        )
    )
    return c


@C.funcs
def geo_effectscatter() -> Geo:
    c = (
        Geo()
        .add_schema(maptype="china")
        .add(
            "城市",
            provin_list,
            type_=ChartType.EFFECT_SCATTER,
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(title_opts=opts.TitleOpts(title="微博城市热度分析"))
    )
    return c


@C.funcs
def geo_heatmap() -> Geo:
    c = (
        Geo()
        .add_schema(maptype="china")
        .add(
            "城市",
            provin_list,
            type_=ChartType.HEATMAP,
        )
        .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(),
            title_opts=opts.TitleOpts(title="微博城市热度分析"),
        )
    )
    return c


# @C.funcs
# def geo_guangdong() -> Geo:
#     c = (
#         Geo()
#         .add_schema(maptype="广东")
#         .add(
#             "geo",
#             [list(z) for z in zip(Faker.guangdong_city, Faker.values())],
#             type_=ChartType.HEATMAP,
#         )
#         .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
#         .set_global_opts(
#             visualmap_opts=opts.VisualMapOpts(),
#             title_opts=opts.TitleOpts(title="Geo-广东地图"),
#         )
#     )
#     return c


# @C.funcs
# def geo_lines() -> Geo:
#     c = (
#         Geo()
#         .add_schema(maptype="china")
#         .add(
#             "",
#             [("广州", 55), ("北京", 66), ("杭州", 77), ("重庆", 88)],
#             type_=ChartType.EFFECT_SCATTER,
#             color="white",
#         )
#         .add(
#             "geo",
#             [("广州", "上海"), ("广州", "北京"), ("广州", "杭州"), ("广州", "重庆")],
#             type_=ChartType.LINES,
#             effect_opts=opts.EffectOpts(
#                 symbol=SymbolType.ARROW, symbol_size=6, color="blue"
#             ),
#             linestyle_opts=opts.LineStyleOpts(curve=0.2),
#         )
#         .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
#         .set_global_opts(title_opts=opts.TitleOpts(title="Geo-Lines"))
#     )
#     return c
#
#
# @C.funcs
# def geo_lines_background() -> Geo:
#     c = (
#         Geo()
#         .add_schema(
#             maptype="china",
#             itemstyle_opts=opts.ItemStyleOpts(color="#323c48", border_color="#111"),
#         )
#         .add(
#             "",
#             [("广州", 55), ("北京", 66), ("杭州", 77), ("重庆", 88)],
#             type_=ChartType.EFFECT_SCATTER,
#             color="white",
#         )
#         .add(
#             "geo",
#             [("广州", "上海"), ("广州", "北京"), ("广州", "杭州"), ("广州", "重庆")],
#             type_=ChartType.LINES,
#             effect_opts=opts.EffectOpts(
#                 symbol=SymbolType.ARROW, symbol_size=6, color="blue"
#             ),
#             linestyle_opts=opts.LineStyleOpts(curve=0.2),
#         )
#         .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
#         .set_global_opts(title_opts=opts.TitleOpts(title="Geo-Lines-background"))
#     )
#     return c


Page().add(*[fn() for fn, _ in C.charts]).render(u'./map.html')
# print([list(z) for z in zip(Faker.provinces, Faker.values())])
# print(Faker.provinces)
# print(type(Faker.provinces))
