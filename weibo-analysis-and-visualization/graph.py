import json

from example.commons import Collector
from pyecharts import options as opts
from pyecharts.charts import Graph, Page

from snapshot_selenium import snapshot as driver
from pyecharts.render import make_snapshot

# C = Collector()
#
# @C.funcs
# def graph_base() -> Graph:
#     nodes = [
#         {"name": "5044281310", "symbolSize": 6},
#         {"name": "3269548101", "symbolSize": 7},
#         {"name": "1267454277", "symbolSize": 5},
#         {"name": "2378564111", "symbolSize": 7},
#         {"name": "1731986465", "symbolSize": 6},
#         {"name": "2280952361", "symbolSize": 7},
#         {"name": "1864135524", "symbolSize": 4},
#         {"name": "2028810631", "symbolSize": 5},
#         {"name": "1737737970", "symbolSize": 6},
#         {"name": "1750167051", "symbolSize": 7},
#         {"name": "1699540307", "symbolSize": 5}
#     ]
#     links = [
#                 {"source": "5044281310", "target": "3269548101"},
#                 {"source": "5044281310", "target": "1017696031"},
#                 {"source": "5044281310", "target": "5687983559"}
#     ]
#     c = (
#         Graph()
#         .add("", nodes, links, repulsion=8000)
#         .set_global_opts(title_opts=opts.TitleOpts(title="Graph-基本示例"))
#     )
#     return c


def graph_weibo() -> Graph:
    with open("weibo.json", "r", encoding='utf-8') as f:
        j = json.load(f)
        nodes, links, categories = j
        print(len(nodes))
        nodes = nodes[:2000]

        # print(nodes)
    c = (
        Graph()
        .add(
            "",
            nodes,
            links,
            categories,
            repulsion=50,
            linestyle_opts=opts.LineStyleOpts(curve=0.2),
            label_opts=opts.LabelOpts(is_show=False),
        )
        .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=False),
            title_opts=opts.TitleOpts(title="Graph-微博评论关系图"),
        )
    )
    return c


graph_weibo().render(u'./graph.html')
# 也可以生成图片，不过效果不好
make_snapshot(driver, graph_weibo().render(), "graph.png")
