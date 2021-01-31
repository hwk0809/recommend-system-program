
import time
import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn import metrics
import copy

# 计时
logfile1 = open("logfile_1.txt",'w')
logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' begin one part\n')

user_sim_mat = pd.read_csv('user_sim_mat_1000.csv',index_col=[0])
user_list=list(user_sim_mat.index)
small_item_list=pd.read_csv('small_item_list.csv')
item_list = list(small_item_list.loc[:,'0'])

train=pd.read_csv('small_train.csv')

# 定义取top函数，返回索引值
def Get_List_Max_Index(list_, n):
    N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
    return list(N_large.index)[:n]

# 生成权值矩阵
def item_weight_list(user1):
    user_sim_list=list(user_sim_mat.loc[user1]) 
    sim_user_columns=Get_List_Max_Index(user_sim_list, 41)# 此处n值为相似用户数
    sim_user_list=[]
    for i in sim_user_columns:
        sim_user_list.append(user_list[i])
    def item_weight(item):
        def user_weight(user2):
            user_data=train.loc[train.user_id==user2]
            user_item=list(set(user_data.item_id))
            if ((item in user_item)&(user1!=user2)):
                part_weight=user_sim_mat.loc[[user1],[user2]]
                return part_weight
            else:
                return 0
        weight=0
        for a in sim_user_list:
            weight+=user_weight(a)
        return weight
    weight_list=list(map(item_weight,item_list))
    return weight_list
weight_mat=list(map(item_weight_list,user_list))
weight_mat_df = pd.DataFrame(weight_mat)
weight_mat_df.index = user_list
weight_mat_df.columns = item_list
weight_mat_df.to_csv('weight_mat.csv',index=True)
print("weight_mat输出完成")

# 进行推荐
def item_suggest(user):
    wt_list=list(weight_mat_df.loc[user])
    suggest_item_columns=Get_List_Max_Index(wt_list, 5)# 此处n值为相似用户数
suggest_mat=list(map(item_suggest,user_list))
suggest_mat_df = pd.DataFrame(suggest_mat)
suggest_mat_df.to_csv('suggest_mat.csv',index=True)
print("suggest_mat输出完成")

logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' finished one part')
logfile1.close()