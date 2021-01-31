

import time
import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn import metrics
import copy

# _______________________________第一部分 计算用户距离_______________________________________

# 计时
logfile1 = open("logfile_1.txt",'w')
logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' begin one part\n')

# 读入数据
LD= pd.read_csv('zuoxiajiao_df2.csv',index_col=[0])
RD= pd.read_csv('youxiajiao_df2.csv',index_col=[0])
LU= pd.read_csv('zuoshangjiao_df2.csv',index_col=[0])
RU= pd.read_csv('youshangjiao_df2.csv',index_col=[0])

#预处理数据
Down=pd.concat([LD,RD],axis=1)
Left=pd.concat([LD,LU])
user_id_list=list(Left.index)

#print (user_id_list)
def cal_sim_list(user1):
    def cal_user_sim(user2):
        def distance(vector1,vector2):  
            d=0
            for a,b in zip(vector1,vector2):  
                d+=(a-b)**2
            return d**0.5
        vec_1=list(Left.loc[user1])
        vec_2=list(Left.loc[user2])
        dist=distance(vec_1,vec_2)
        user_sim=1/(1+dist)
        print('1')
        return user_sim
    user_sim_list=list(map(cal_user_sim,user_id_list))
    return user_sim_list
user_sim_mat=list(map(cal_sim_list,user_id_list))
user_sim_df = pd.DataFrame(user_sim_mat)
user_sim_df.index = user_id_list
user_sim_df.columns = user_id_list
user_sim_df.to_csv('user_sim_mat_2.csv',index=True)
print("输出完成")
logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' finished one part')
logfile1.close()