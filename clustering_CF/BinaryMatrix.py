# -*- coding:utf-8 -*-
"""
作者: hwk
日期: 2020年12月23日
"""
import numpy as np
import pandas as pd

# 将矩阵转为二值化矩阵，小于a的元素设置为0
def Binary(mat,a):
    mat_list = []
    m = np.shape(mat)[0]
    n = np.shape(mat)[1]
    for i in range(m):
        temp = []
        for j in range(n):
            if mat[i,j]<a:
                temp.append(0)
            else:
                temp.append(1)
        mat_list.append(temp)
    mat_binary = np.mat(mat_list)
    return mat_binary




if __name__ == "__main__":
    a = 1
    # 小于1/4的设为0，查找其数值
    # df = pd.read_csv('userid_cateid_new_norm.csv', index_col=0)
    # df = df.values
    # m = np.shape(df)[0]
    # n = np.shape(df)[1]
    # a = []
    # for i in range(m):
    #     for j in range(n):
    #         a.append(df[i,j])
    # b=list(set(a))
    # k = len(b)
    # print(k)
    # print(k/4)
    # print(b[153])  #0.34

    # a= 0.34
    # matix = np.mat([[0.3,0.1,0.1],[0.4,0.6,0.8]])
    # b=Binary(matix, a)
    # print(b)

    # dict1 = {'a': 2, 'e': 3, 'f': 8, 'd': 4}
    # name = {'a','e','f','d'}
    # Df = [[1,2,3],[2,3,4],[4,6,7],[0,1,2]]
    # df = pd.DataFrame(Df,index=name)
    # print(df)
    # # print(df.loc['a'].tolist())
    # dict2 = sorted(dict1,reverse = True)
    # print(dict2)
    # k=2
    # leader_r =[]
    #
    # for i in range(k):
    #     leader = dict2[i]
    #     leader_r.append(df.loc[leader].tolist())
    # print(leader_r)
    #
    # cluster = km.clustering_kmeans(df,k,np.array(leader_r))