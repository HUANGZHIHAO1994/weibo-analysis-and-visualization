import json
import os
import pandas as pd
from example.commons import Collector
from pyecharts import options as opts
from pyecharts.charts import Page, Tree


df_weibo = pd.read_csv('LDA_total.csv', sep=',', encoding='gb18030')
df_weibo = df_weibo.drop(['主题数标号'], axis=1)
GroupBy = df_weibo.groupby(['关键词'])
data = []

# 每个关键词的树
data2 = []
count = 1
for name, group in GroupBy:
    # print(name)
    l = []
    d2 = {}
    # count += 1
    # if (count >= 6) and (name != "贸易战"):
    #     continue
    for i in group['主题']:
        d = {}
        d["name"] = i
        l.append(d)
        # print(group['主题'])
    d2["children"] = l
    d2["name"] = name
    data2.append(d2)

#  最外层树
dic2 = {}
dic2["children"] = data2
dic2["name"] = "关键词"
data.append(dic2)
print(data)

# 需要调整画布大小
tree = Tree(init_opts=opts.InitOpts(width='2000px', height='15000px'))
# 两个树分支的距离增大
tree.add("", data, collapse_interval=100)
tree.set_global_opts(title_opts=opts.TitleOpts(title="LDA主题模型"))
tree.render(u'./tree.html')


# 用如下程序也行
# C = Collector()
#
#
# @C.funcs
# def tree_base() -> Tree:
#
#     c = (
#         Tree()
#         .add("", data, collapse_interval=100)
#         # ._set_collapse_interval(10)
#         .set_global_opts(title_opts=opts.TitleOpts(title="LDA主题模型"))
#     )
#     return c


# @C.funcs
# def tree_lr() -> Tree:
#     with open(os.path.join("fixtures", "flare.json"), "r", encoding="utf-8") as f:
#         j = json.load(f)
#     c = (
#         Tree()
#         .add("", [j], collapse_interval=2)
#         .set_global_opts(title_opts=opts.TitleOpts(title="Tree-左右方向"))
#     )
#     return c
#
#
# @C.funcs
# def tree_rl() -> Tree:
#     with open(os.path.join("fixtures", "flare.json"), "r", encoding="utf-8") as f:
#         j = json.load(f)
#     c = (
#         Tree()
#         .add("", [j], collapse_interval=2, orient="RL")
#         .set_global_opts(title_opts=opts.TitleOpts(title="Tree-右左方向"))
#     )
#     return c
#
#
# @C.funcs
# def tree_tb() -> Tree:
#     with open(os.path.join("fixtures", "flare.json"), "r", encoding="utf-8") as f:
#         j = json.load(f)
#     c = (
#         Tree()
#         .add(
#             "",
#             [j],
#             collapse_interval=2,
#             orient="TB",
#             label_opts=opts.LabelOpts(
#                 position="top",
#                 horizontal_align="right",
#                 vertical_align="middle",
#                 rotate=-90,
#             ),
#         )
#         .set_global_opts(title_opts=opts.TitleOpts(title="Tree-上下方向"))
#     )
#     return c
#
#
# @C.funcs
# def tree_bt() -> Tree:
#     with open(os.path.join("fixtures", "flare.json"), "r", encoding="utf-8") as f:
#         j = json.load(f)
#     c = (
#         Tree()
#         .add(
#             "",
#             [j],
#             collapse_interval=2,
#             orient="BT",
#             label_opts=opts.LabelOpts(
#                 position="top",
#                 horizontal_align="right",
#                 vertical_align="middle",
#                 rotate=-90,
#             ),
#         )
#         .set_global_opts(title_opts=opts.TitleOpts(title="Tree-下上方向"))
#     )
#     return c
#
#
# @C.funcs
# def tree_layout() -> Tree:
#     with open(os.path.join("fixtures", "flare.json"), "r", encoding="utf-8") as f:
#         j = json.load(f)
#     c = (
#         Tree()
#         .add("", [j], collapse_interval=2, layout="radial")
#         .set_global_opts(title_opts=opts.TitleOpts(title="Tree-Layout"))
#     )
#     return c


# Page().add(*[fn() for fn, _ in C.charts]).render(u'./tree.html')
