import time
import pandas as pd
import numpy as np
import re
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import re
import math
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer
import Clustering_kmeans as km
import Normalization_df as Norm
import Clusering_evaluate as EV

new_mt = pd.read_csv('new_mat_norm.csv', index_col=0)
print(new_mt)

# # '利用SSE选择k'
SSE = []  # 存放每次结果的误差平方和
for k in range(3, 30):
    estimator = KMeans(n_clusters=k)  # 构造聚类器
    estimator.fit(new_mt.values)
    SSE.append(estimator.inertia_)
    print(SSE[k-3])
X = range(3, 30)
plt.xlabel('k')
plt.ylabel('SSE')
plt.plot(X, SSE, 'o-')
plt.show()

# Scores = []  # 存放轮廓系数
# for k in range(3,30):
#     labels=km.clustering_kmeans_predict(new_mt,k,new_mt )
#     Scores.append(EV.cal_silhouette(new_mt.values,labels))
#     print(Scores[k])
# print(Scores.index(max(Scores)))
# X = range(3,30)
# plt.xlabel('k')
# plt.ylabel('轮廓系数')
# plt.plot(X, Scores, 'o-')
# plt.show()

# # 训练数据聚类
# k = 3
# km = KMeans(n_clusters=k)
# K = km.fit((user_cate_df/3).values)
# train_label = km.labels_
#
# # 分类器用于test数据
# test_label = km.predict(test_user_cate_df.values)
# print(train_label)        # 训练集的聚类标签
# print(test_label)        # test聚类标签