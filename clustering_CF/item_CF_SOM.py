# -*- coding:utf-8 -*-
"""
作者: yinboer
日期: 2020年12月23日
"""
import pandas as pd
import numpy as np
import re
import math
from operator import itemgetter
from sklearn.feature_extraction import DictVectorizer
import Clustering_SOM as SO
import Clustering_kmeans as km
from sklearn import metrics
import time
import Normalization_df as Norm
import matplotlib.pyplot as plt
import Clusering_evaluate as EV
import plot_cluster_multidimensional as PM

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
        print(u)
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
                    list1.append(i)
                    list2.append(j)
        for m in range(len(list1)):
            count = 0
            for cate, cate_items in cluster.items():
                if int(list1[m]) in cate_items and int(list2[m]) in cate_items:
                    count = 1
                    break
            if count == 0:
                del C[list1[m]][list2[m]]

    print("稀疏矩阵: ", C)
    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W.setdefault(i, {})
            W[i].setdefault(j, 0)
            W[i][j] = cij / math.sqrt(N[i] * N[j]) * 100

    print("物品相似度: ", W)
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

# 计算DCG指数
def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg

def cal_4_index(U, T):
    T=np.array(T)
    N=2
    U_MAE=[]
    U_prec=[]
    U_recall=[]
    U_DCG=[]
    recommend_N = list(np.array(U).argsort()[::-1][0:10])
    trans_index = [sum(T>=2)+1,sum(T>=3)+1,sum(T>=4)+1,sum(T>=5)+1,1]
    while N<=10:
        U_MAE.append(metrics.mean_absolute_error(T, U))
        U_prec.append(sum(T[recommend_N[:N]]>=4)/min(N,len(T)))
        U_recall.append(sum(T[recommend_N[:N]]>=4)/sum(T>=4))
        true_rank = [trans_index[T[i]-1] for i in recommend_N[:N]]
        U_DCG.append(getNDCG(true_rank, list(range(1, min(len(T),N)+1))))
        N = N+1
    print([U_MAE, U_prec, U_recall, U_DCG])
    return [U_MAE, U_prec, U_recall, U_DCG]

# 载入数据
LD = pd.read_csv('zuoxiajiao_df.csv', index_col=0)
RD = pd.read_csv('youxiajiao_df.csv', index_col=0)
LU = pd.read_csv('zuoshangjiao_df.csv', index_col=0)
RU = pd.read_csv('youshangjiao_df.csv', index_col=0)

# 预处理数据
Down = pd.concat([LD, RD], axis=1)
Down = Down.head(10)
train = DF_to_dict(Down)
test = DF_to_dict(LU)
#
# # 生成物品向量
# user_id_list = Down.index.values.tolist()
# user_items, column_names = count_items(train)
# mt = pd.DataFrame(user_items, index=user_id_list, columns=column_names)
# new_mt = mt.T
#

# # 聚类前置数据处理
# new_mt.to_csv('new_mat.csv',float_format='%.4f')
# new_mt = Norm.normalization(new_mt)
# new_mt.to_csv('new_mat_norm.csv',float_format='%.4f')
new_mt =pd.read_csv('new_mat_norm.csv')



# k = 7
# cluster5 = km.clustering_kmeans(new_mt,k)
#SOM
sommat = np.mat(new_mt.values)
som = SO.SOM(sommat,(5,5),1,new_mt.shape[0])
som.train()
res = som.train_result()
cluster3 = {}
for i, win in enumerate(res):
    if not cluster3.get(win[0]):
        cluster3.setdefault(win[0], [i])
    else:
        cluster3[win[0]].append(i)
# print(cluster3)


# 计算物品相似度矩阵，另存为文件
# W = item_similarity(train,cluster5)
# W5_df = pd.DataFrame(W)
# W5_df.to_csv('W_kmeans_7.csv',float_format='%.2f')

W =pd.read_csv('W_SOM_df.csv', index_col=0)
W2 = W.fillna(0)
new_W = DF_to_dict(W2)

# 计算
time_start=time.time()
test_userid = RU.index.values.tolist()
expected_itemid = RU.columns.values.tolist()
predict_rating = {}
all_pred_label = []
all_true_label = []
index_all = []
for userid in test_userid:
    sub_pred_label = []
    sub_true_label=[]
    print(1)
    for itemid in expected_itemid:
        a=RU.iloc[test_userid.index(userid)][itemid]
        if (a==0):
            continue
        b = calc_rating(userid, itemid, test, new_W)
        sub_pred_label.append(b)
        sub_true_label.append(a)
        all_pred_label.append(b)
        all_true_label.append(a)
    index_all.append(cal_4_index(sub_pred_label, sub_true_label))
time_end=time.time()
print('totally cost',time_end-time_start)
np.save('cluster_SOM1',index_all)





