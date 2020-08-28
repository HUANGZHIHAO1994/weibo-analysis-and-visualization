<div align="left">
    <img src='https://ftp.bmp.ovh/imgs/2020/08/b77a8439ea51e080.jpg' height="50" width="50" >
 </div>

![weibo](https://badgen.net/badge/weibo/NLP/cyan?icon=github)
![GitHub license](https://badgen.net/github/license/HUANGZHIHAO1994/weibo-analysis-and-visualization?color=green)
![stars](https://badgen.net/github/stars/HUANGZHIHAO1994/weibo-analysis-and-visualization)
![forks](https://badgen.net/github/forks/HUANGZHIHAO1994/weibo-analysis-and-visualization?color=red)
![python](https://badgen.net/badge/python/%3E=3.6/8d6fe7)

# 微博文本分析和可视化


## 0.  数据来源和结构

新浪微博，爬虫链接：

[https://github.com/HUANGZHIHAO1994/weibospider-keyword](https://github.com/nghuyong/WeiboSpider)

微博内容数据结构（mongo数据库导出的json文档）

```
content_example:
[
{'_id': '1177737142_H4PSVeZWD', 'keyword': 'A股', 'crawl_time': '2019-06-01 20:31:13', 'weibo_url': 'https://weibo.com/1177737142/H4PSVeZWD', 'user_id': '1177737142', 'created_at': '2018-11-29 03:02:30', 'tool': 'Android', 'like_num': {'$numberInt': '0'}, 'repost_num': {'$numberInt': '0'}, 'comment_num': {'$numberInt': '0'}, 'image_url': 'http://wx4.sinaimg.cn/wap180/4632d7b6ly1fxod61wktyj20u00m8ahf.jpg', 'content': '#a股观点# 鲍威尔主席或是因为被特朗普总统点名批评后萌生悔改之意，今晚一番讲话被市场解读为美联储或暂停加息步伐。美元指数应声下挫，美股及金属贵金属价格大幅上扬，A50表现也并不逊色太多。对明天A股或有积极影响，反弹或能得以延续。 [组图共2张]'},...
]
```

微博评论数据结构（mongo数据库导出的json文档）

```
comment_example:
[
{'_id': 'C_4322161898716112', 'crawl_time': '2019-06-01 20:35:36', 'weibo_url': 'https://weibo.com/1896820725/H9inNf22b', 'comment_user_id': '6044625121', 'content': '没问题，', 'like_num': {'$numberInt': '0'}, 'created_at': '2018-12-28 11:19:21'},...
]
```



## 1.  数据预处理

1. **prepro.py、pre_graph.py、senti_pre.py**

   为了应对各种分析需求，需要数据预处理，具体所需数据文件类型和输出的结果数据结构见这三个py文件

   PS：

   **prepro.py**  运行时根据需要修改123、143、166行三处代码

   **pre_graph.py**  运行时根据需要修改127、140行两处代码

   **senti_pre.py**  运行时根据需要修改第119行代码

2. **zh_wiki.py、langconv.py**  

   这两个py文件是用于繁体转简体的无需修改

# 2.  数据分析和可视化

1. **词云：wc.py**（需要跑完prepro.py）

   根据需要修改3、19、26行代码

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/a5905208795f2ac7.png?raw=true'
            >
   </div>

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/fa51683f710a6473.png?raw=true'         
            >
   </div>

   

2. **热度地图：** **map.py**（需要跑完prepro.py）

   根据需要修改第8行代码

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/50a61c72f949a0b9.png?raw=true'         
            >
   </div>

   

3. **转发、评论、点赞时间序列：** **line.py**（需要跑完senti_pre.py 和 senti_analy.py）

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/450a55ff983db14a.png?raw=true'
            >    
   </div>

   

4. **微博评论关系图：** **graph.py**（需要跑完pre_graph.py）

   （[参考](https://blog.csdn.net/Kevin_HZH/article/details/91043392)）

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/6848edc9ac9a4a5a.png?raw=true'         
            >
   </div>

   

5. **文本聚类：** **cluster_tfidf.py** 和 **cluster_w2v.py**（需要跑完prepro.py）

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/6981da3109f690ac.png?raw=true'         
            >
   </div>

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/83226f9c65632680.png?raw=true'         
            >
   </div>

   

6. **LDA主题模型分析：** **LDA.py**（需要跑完senti_pre.py）**tree.py**（需要跑完senti_analy.py）

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/7f5d68f1397c3732.png?raw=true'         
            >
   </div>

   

7. **情感分析（词典）：** **senti_analy.py**（需要跑完senti_pre.py）**3Dbar.py**（需要跑完senti_analy.py）**pie.py**（需要跑完senti_analy.py）

   <div>
       <img
   src='https://ftp.bmp.ovh/imgs/2020/08/fc6e429690f5db99.png?raw=true'         
            >
   </div>

   

8. **情感分析（W2V+LSTM）：Sentiment-Analysis-master文档中的senti_lstm.py**（需要跑完senti_pre.py）

   看情况修改250行代码

   有些文档太大，放在百度网盘链接中:

   链接：https://pan.baidu.com/s/1l447d3d6OSd_yAlsF7b_mA 
   提取码：og9t

   


9. **文本相似度分析：similar.py**（仅供参考）

   

10. **其他可供参考：**  **senti_analy_refer.py、Sentiment_lstm.py**

    

11. **有关Senti_Keyword_total_id.csv：**  

    下载8.百度网盘中Senti_Keyword_total_id.csv即可，以下是解释：
    该文件几乎和Senti_Keyword_total.csv相同，只是多了一列weibo_id（此处不再给出生成Senti_Keyword_total_id.csv的代码，直接给生成的文档，
    生成Senti_Keyword_total_id.csv可改写**senti_analy.py**，增加一列weibo_id），
    8中的百度网盘（有Senti_Keyword_total_id.csv和Senti_Keyword_total.csv，还有全部comment和全部content），
    由于lines.py等需要全部关键词，因此需要用**senti_analy.py**直接跑全部comment.json和content.json生成Senti_Keyword_total.csv（直接从网盘下来Senti_Keyword_total_id.csv再跑lines.py，3Dbar.py，pie.py即可）













