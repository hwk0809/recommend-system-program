# -*- coding:utf-8 -*-
"""
作者: hwk
日期: 2020年12月22日
"""
import numpy as np
import pandas as pd

# 先中心化，再正则化
def normalization(df):
    rowname = df._stat_axis.values.tolist()
    columnname = df.columns.values.tolist()
    mat = np.mat(df.values)
    m = np.shape(mat)[0]  # 行数
    n = np.shape(mat)[1]  # 列数
    mat_norm = []
    for i in range(m):
        list =[]
        temp = np.linalg.norm(mat[i,:])
        avg = 0
        if temp == 0:
            mat_norm = mat_norm.append(mat[i,:].tolist()[0])
            continue
        for j in range(n):
            avg = avg + mat[i,j]
        avg = avg / n
        for j in range(n):
            t = mat[i,j]/temp-avg
            list.append(t)
        mat_norm.append(list)
    mat_norm = np.array(mat_norm)
    df_norm = pd.DataFrame(mat_norm,columns=columnname,index=rowname)
    return df_norm


if __name__ == "__main__":
    user_cate_df = pd.read_csv('userid_cateid_new.csv', index_col=0)
    a =np.mat([[0,1],[1,2],[3,4]])
    df = pd.DataFrame(a)
    print(df)
    # print(df.values)
    # # mat =df.values
    #     # mat = np.mat(df)
    #     # i = 1
    #     # mat[i, :] = mat[i, :] / 0.1
    #     # print(mat)
    df_norm =normalization(df)
    df_norm = normalization(user_cate_df)
    print(df_norm)
    df_norm.to_csv('userid_cateid_new_norm.csv')
