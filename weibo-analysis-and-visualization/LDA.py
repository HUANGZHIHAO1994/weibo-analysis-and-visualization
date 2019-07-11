'''
使用senti_pre.py形成的content2.pkl进行LDA主题模型分析，生成'LDA_total.csv'
为tree.py做准备
'''

from jieba import analyse
import jieba
from gensim import corpora, models
import pickle
import csv

jieba.add_word('中国')

f = open("./key_words2.txt", 'r', encoding='UTF-8-sig')
s = f.read()
s = s.replace('\n', '；')
s = s.replace(' ', '')
f.close()

start_uids1 = s.split('；')[:-1]
start_uids = list(set(start_uids1))
start_uids.sort(key=start_uids1.index)

content_comment = pickle.load(open('./content2.pkl', 'rb'))

out = open('LDA_total.csv', 'a', newline='', encoding='gb18030')
csv_write = csv.writer(out, dialect='excel')
stu1 = ['关键词', '主题数标号', '主题']
csv_write.writerow(stu1)

for uid in start_uids:
    uid = uid.strip()
    count = 1
    textrank = []
    for i in content_comment:
        if i[3].strip() == uid:
            print("{}处理到：".format(uid), count)
            count += 1

            word_textrank = []
            for x, w in analyse.textrank(i[1], topK=5, withWeight=True):
                word_textrank.append(x)
            textrank.append(word_textrank)

            # 做映射，相当于词袋，字典形式
    try:
        dictionary = corpora.Dictionary(textrank)  # 输入是list of list格式，每篇文章分词把文章集合起来再形成list
        corpus = [dictionary.doc2bow(sentence) for sentence in textrank]

        lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=10)

        for topic in lda.print_topics(num_topics=10, num_words=5):
            lda_1 = []
            lda_1.append(uid)
            lda_1.append(topic[0])
            lda_1.append(topic[1])

            csv_write.writerow(lda_1)
    except:
        continue

