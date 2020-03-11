import time          
import re          
import os  
import sys
import codecs
import shutil
import numpy as np
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
###############################First Step: TF-IDF###########################################                           第一步 计算TFIDF

#文档预料 空格连接
corpus = []
#读取预料 一行预料为一个文档
with open("neg_training1.txt", "r") as f:
    corpus = list(f.readlines())
#for line in open('neg_training.txt', 'r').readlines():
#     print(line)
#	corpus.append(line.strip())
# print(corpus)
#time.sleep(1)
#将文本中的词语转换为词频矩阵 矩阵元素a[i][j] 表示j词在i类文本下的词频
vectorizer = CountVectorizer()

#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()

#第一个fit_transform是计算tf-idf 第二个fit_transform是将文本转为词频矩阵
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))
print(tfidf)
#获取词袋模型中的所有词语  
word = vectorizer.get_feature_names()

print("begin to train tfidf")
#将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
weight = tfidf.toarray()

print("finish training tfidf")
#打印特征向量文本内容
# print('Features length: ' + str(len(word)))
# resName = "BHTfidf_Result.txt"
# result = codecs.open(resName, 'w', 'utf-8')
# for j in range(len(word)):
#     result.write(word[j] + ' ')
# result.write('\r\n\r\n')

#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重  
# for i in range(len(weight)):
# #     print u"-------这里输出第",i,u"类文本的词语tf-idf权重------"  
#     for j in range(len(word)):
#         #print weight[i][j],
#         result.write(str(weight[i][j]) + ' ')
#     result.write('\r\n\r\n')

# result.close()

################################Second Step : Cluster####################################                               第二步 聚类Kmeans

print('Start Kmeans:')
from sklearn.cluster import KMeans
best_cluster = 0
least_distance = 100000
result = []
for i in range(3,6):#选择最佳的簇数
	clf = KMeans(n_clusters=i)
	s = clf.fit(weight)
	#20个中心点
#     print(clf.cluster_centers_)
	#每个样本所属的簇
	print(clf.labels_)
	#用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
	if(clf.inertia_ < least_distance):
		least_distance = clf.inertia_
		best_cluster = i
		result = clf.labels_
print(best_cluster)
print(result)
f1=open('neg_clusters1.txt','w')
f1.write(str(result))
f1.close()
