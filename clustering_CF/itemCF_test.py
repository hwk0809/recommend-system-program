# -*- coding:utf-8 -*-
"""
作者: yinboer
日期: 2020年12月22日
"""
import pandas as pd
import numpy as np
import re
import math
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer
import Clustering_kmeans as km
import Clustering_DBSCN as DB
import Clustering_SOM as SO
import Clusering_evaluate as EV
import matplotlib.pyplot as plt

# 化为字典形式，并删除值为零的键值对
def DF_to_dict(dataframe):
    new_dataframe = dataframe.T
    dict = new_dataframe.to_dict()
    list1 = []
    list2 = []
    for i,n in dict.items():
        for j,k in n.items():
            list1.append(i)
            list2.append(j)
    for m in range(len(list1)):
        if dict[list1[m]][list2[m]] == 0:
            del dict[list1[m]][list2[m]]

    return dict

# 构造物品标签向量阵
def count_items(dict):
    restructured = []
    for key in dict:
        data_dict = {}
        for news in dict[key]:
            data_dict[news] = 1
        restructured.append(data_dict)
    dictvectorizer = DictVectorizer(sparse=False)
    features = dictvectorizer.fit_transform(restructured)
    columns_names = dictvectorizer.get_feature_names()
    return features, columns_names

# 计算稀疏矩阵与物品相似度
def item_similarity(userSet, cluster = None):
    C = dict()
    N = dict()
    for u, items in userSet.items():
        for i in items:
            N.setdefault(i, 0)
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                C.setdefault(i, {})
                C[i].setdefault(j, 0)
                C[i][j] += 1 / math.log(1 + len(items))

    # 聚类
    list1 = []
    list2 = []
    if cluster != None:
        for i, related_j in C.items():
            for j, similarity in related_j.items():
                count = 0
                for cate, cate_items in cluster.items():
                    if i in cate_items and j in cate_items:
                        count = 1
                        break
                if count == 0:
                    list1.append(i)
                    list2.append(j)
        for m in range(len(list1)):
                del C[list1[m]][list2[m]]

    # print("稀疏矩阵: ", C)
    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W.setdefault(i, {})
            W[i].setdefault(j, 0)
            W[i][j] = cij / math.sqrt(N[i] * N[j]) * 100

    # print("物品相似度: ", W)
    return W


def calc_rating(user_id, item_id, test, W):
    rank = dict()
    interacted_items = test[user_id]  # 该用户的浏览情况
    # print(interacted_items)
    for item in interacted_items: # 根据该资讯寻找相似资讯
        if int(item) not in W.keys():
            continue  # 若该资讯无相似资讯则跳过？
        related_item = []
        for re_item, score in W[int(item)].items():
            related_item.append((re_item, score))

        for j, v in sorted(related_item, key=itemgetter(1), reverse=True):  # j:资讯名；v：该资讯权重；  根据权重从大到小排序
            # print(j, ":", v)
            if j in interacted_items:
                continue  # 若用户浏览过该资讯则跳过
            if j not in rank.keys():
                rank[j] = 0  # 若该咨询第一次出现则初始化
            pi = interacted_items[item]
            rank[j] += pi * v
    if rank:
        if item_id in rank:
            rank_max = max(rank.values())
            rank_min = min(rank.values())
            for key in rank.keys():
                rank[key] = (rank[key] - rank_min) / (rank_max - rank_min) * 5
            result = rank[item_id]
        else:
            result = 0
    else:
        result = 0
    return result

# 载入数据
LD = pd.read_csv('zuoxiajiao_df.csv', index_col=0)
RD = pd.read_csv('youxiajiao_df.csv', index_col=0)
LU = pd.read_csv('zuoshangjiao_df.csv', index_col=0)
RU = pd.read_csv('youshangjiao_df.csv', index_col=0)

# 预处理数据
Down = pd.concat([LD, RD], axis=1)
# Down = Down.head(483)
train = DF_to_dict(Down)
test = DF_to_dict(LU)

# 生成物品向量
user_id_list = Down.index.values.tolist()
user_items, column_names = count_items(train)
mt = pd.DataFrame(user_items,index=user_id_list,columns=column_names)
new_mt = mt.T

#kmeans
Scores = []
for k in range(2,5):
    labels=km.clustering_kmeans_predict(new_mt,k,new_mt)
    Scores.append(EV.cal_silhouette(new_mt.values,labels))
print(Scores)
print(Scores.index(max(Scores)))
X = range(2,5)
plt.xlabel('k')
plt.ylabel('轮廓系数')
plt.plot(X, Scores, 'o-')
plt.show()
# k = 19
# cluster1 =km.clustering_kmeans(new_mt,k)
# print(cluster1)

# 计算物品相似度矩阵，另存为文件
# W = item_similarity(train,cluster)
#
# W_df = pd.DataFrame(W)
# W_df.to_csv('W_df.csv',float_format='%.2f')

# W =pd.read_csv('W_df.csv', index_col=0)
# W2 = W.fillna(0)
# new_W = DF_to_dict(W2)


# 计算
# test_userid = RU.index.values.tolist()
# expected_itemid = RU.columns.values.tolist()
# predict_rating = {}
# for userid in test_userid:
#     for itemid in expected_itemid:
#         predict_rating.setdefault(userid, {})
#         predict_rating[userid][itemid] = calc_rating(userid, itemid, test, W)
# pr_df =pd.DataFrame(predict_rating)
# pr_df.index = test_userid
# pr_df.columns = expected_itemid
# pr_df.to_csv('pr_df.csv',float_format='%.2f')
#
# rating = calc_rating(52, '35', test, W)
# print(rating)
