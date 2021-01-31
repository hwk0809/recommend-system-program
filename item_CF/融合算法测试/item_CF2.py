# -*- coding:utf-8 -*-
"""

作者: hwk
日期: 2020年12月17日
"""
import pandas as pd
import re
import math
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer
import Clustering_kmeans as km


# 载入内容标签数据
def load_itemdata(Path):
    items = pd.read_csv(Path)
    items_id_list = list(set(items.movieId))
    dataSet = {}
    for index, row in items.iterrows():# 对行遍历
        item = str(row['movieId'])
        dataSet.setdefault(item, {})
        dataSet[item] = row['genres'].split('|')
    # print(dataSet)
    return dataSet, items_id_list

# 构造物品标签向量阵
def count_tags(dict):
    restructured = []
    for key in dict:
        data_dict = {}
        for news in dict[key]:
            data_dict[news] = 1
        restructured.append(data_dict)
    dictvectorizer = DictVectorizer(sparse=False)
    features = dictvectorizer.fit_transform(restructured)
    columns_names = dictvectorizer.get_feature_names()
    return features,columns_names

# 建立标签-内容的倒排列表
def calc_tag_item(dataSet):
    tag_items = dict()
    for items, tags in dataSet.items():
        for tag in tags:
            if tag not in tag_items:
                tag_items[tag] = set()
            tag_items[tag].add(items)
    # print(tag_items)
    return tag_items


#
#     item_tags = {}
#     for item, tags in itemdata.items():
#         item_tags.setdefault(item, {})
#         for tag in tags:
#             if tag not in item_tags[item].keys():
#                 item_tags[item][tag] = 0
#             else:
#                 item_tags[item][tag] += 1
#     print(item_tags)
#     return

# 载入用户评分数据
def load_userdata(Path):
    # 读入用户ID，存入user_id_list列表
    train = pd.read_csv(Path)
    # print(train)
    user_id_list = list(set(train.userId))
    # print(user_id_list)

    dataSet = {}
    for user in user_id_list:
        user_train = train.loc[train.userId == user]  # 该名用户的所有浏览记录
        for index, row in user_train.iterrows():
            dataSet.setdefault(user, {})
            dataSet[user][str(int(row['movieId']))] = row['rating']
        # user_items = list(set(user_train.item_id))  # 该名用户浏览过的所有items
    return dataSet, user_id_list

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
                count = 0
                if cluster != None:
                    for cate, cate_items in cluster.items():
                        if i in cate_items and j in cate_items:
                            count = 1
                            break
                    if count == 0:
                        continue #若i，j不在同一个类里则不计算相似度
                C.setdefault(i, {})
                C[i].setdefault(j, 0)
                C[i][j] += 1 / math.log(1 + len(items))

    # print("稀疏矩阵: ", C)
    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W.setdefault(i, {})
            W[i].setdefault(j, 0)
            W[i][j] = cij / math.sqrt(N[i] * N[j])

    # print("物品相似度: ", W)
    return W

# 推荐
def recommend(user_list, train, W, K):
    rank = dict()
    result = dict()
    for alluser in user_list:
        rank.clear()
        if alluser not in train:
            continue  # 若无该用户浏览记录则跳过？
        interacted_items = train[alluser]  # 该用户的浏览情况

        for item in interacted_items: # 根据该资讯寻找相似资讯
            if item not in W:
                continue  # 若该资讯无相似资讯则跳过？
            # print(">>>>>>>>", item)
            related_item = []
            for re_item, score in W[item].items():
                related_item.append((re_item, score))

            for j, v in sorted(related_item, key=itemgetter(1), reverse=True):  # j:资讯名；v：该资讯权重；  根据权重从大到小排序
                # print(j, ":", v)
                if j in interacted_items:
                    continue  # 若用户浏览过该资讯则跳过
                if j not in rank.keys():
                    rank[j] = 0  # 若该咨询第一次出现则初始化
                pi = interacted_items[item]
                rank[j] += pi * v
        recom = []
        for j, v in sorted(rank.items(), key=itemgetter(1), reverse=True)[0:K]:
            recom.append(j)
        result[alluser] = recom
    return result

#载入数据
item_data, item_id_list = load_itemdata('movies.csv')
user_data, user_id_list = load_userdata('ratings2.csv')

# 生成电影-标签矩阵
movie_tags, column_names = count_tags(item_data)
new_mt = pd.DataFrame(movie_tags,index=item_id_list,columns=column_names)
print(new_mt)

# Scores = []
# for k in range(3,21):
#     labels=km.clustering_kmeans(new_mt,k)
#     Scores.append(EV.cal_silhouette(new_mt.values,labels))
# print(Scores)
# X = range(3,21)
# plt.xlabel('k')
# plt.ylabel('轮廓系数')
# plt.plot(X, Scores, 'o-')
# plt.show()

# 对于物品标签做聚类

#kmeans
k = 19
cluster1 =km.clustering_kmeans(new_mt,k)
print(cluster1)


# DBSCN
# cluster2 = DB.clustering_DBSCAN(new_mt,DistanceFlag=1,r=0.2,min_number=30)
# print(cluster2)

#SOM
# print(new_mt.shape[0])
# som = SO.SOM(new_mt,(5,5),1,new_mt.shape[0])
# som.train()
# res = som.train_result()
# cluster3 = {}
# for i, win in enumerate(res):
#     if not cluster3.get(win[0]):
#         cluster3.setdefault(win[0], [i])
#     else:
#         cluster3[win[0]].append(i)
# print(cluster3)


# tag_data = calc_tag_item(item_data)
user_W = item_similarity(user_data)
# tag_W = item_similarity(tag_data)
user_result = recommend(user_id_list, user_data, user_W, 5)
# tag_result = recommend(user_id_list, user_data, tag_W, 5)
print('ItemCF推荐：', user_result)
# print('基于内容推荐：', tag_result)