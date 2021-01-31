# -*- coding:utf-8 -*-
"""
作者: yinboer
日期: 2020年12月12日
"""
import math
from collections import defaultdict
from operator import itemgetter

def load_data(filePath):
    dataSet = {}
    with open(filePath, "r", encoding="utf-8") as  f:
    	for line in f:
       	    user, hero, rating = line.strip().split(",")
            dataSet .setdefault(user, {})
       	    dataSet [user][hero] = rating
    return dataSet


# def calc_user_sim(dataSet):  # 建立物品-用户的倒排列表
#     item_users = dict()
#     for user, items in dataSet.items():
#         # print(user)
#         # print(items)
#         for movie in items:
#             if movie not in item_users:
#                 item_users[movie] = set()
#             item_users[movie].add(user)
#     return item_users

def calc_item_sim(dataSet):  # 建立用户-物品的倒排列表
    users_item = dict()
    for user, items in dataSet.items():
        if user not in users_item:
            users_item[user] = set()
        for movie in items:
            users_item[user].add(movie)
    return users_item

# def user_similarity(userSet):
#     C = dict()
#     N = dict()
#     for movie, users in userSet.items():
#         for u in users:
#             N.setdefault(u, 0)
#             N[u] += 1
#             for v in users:
#                 if u == v:
#                     continue
#                 C.setdefault(u, {})
#                 C[u].setdefault(v, 0)
#                 C[u][v] += 1 / math.log(1 + len(users))
#
#     print("稀疏矩阵: ", C)
#     W = dict()
#     for u, related_users in C.items():
#         for v, cuv in related_users.items():
#             W.setdefault(u, {})
#             W[u].setdefault(v, 0)
#             W[u][v] = cuv / math.sqrt(N[u] * N[v])
#     print("用户相似度: ", W)
#     return W


def item_similarity(userSet):
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

    print("稀疏矩阵: ", C)
    W = dict()
    for i, related_items in C.items():
        for j, cij in related_items.items():
            W.setdefault(i, {})
            W[i].setdefault(j, 0)
            W[i][j] = cij / math.sqrt(N[i] * N[j])

    print("物品相似度: ", W)
    return W


def recommend(user, train, W, K):
    pi = 1
    rank = dict()
    interacted_items = train[user]

    for item in interacted_items:
        print(">>>>>>>>", item)
        related_item = []
        for user, score in W[item].items():
            related_item.append((user, score))
        print(related_item)

        for j, v in sorted(related_item, key=itemgetter(1), reverse=True)[0:K]:
            print(j, v)
            if j in interacted_items:
                continue
            if j not in rank.keys():
                rank[j] = 0
            rank[j] += pi * v
    print("推荐人物: ", rank)
    return rank



data=load_data('ItemIF_info.txt')
print(data)
# item_users=calc_user_sim(data)
# print("物品-用户倒排列表: ", item_users)
# user_similarity(item_users)
users_item = calc_item_sim(data)
print("用户-物品倒排列表：",users_item)
W = item_similarity(data)
recommend('A',data,W,3)
