

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

# 读入user_port
cate_num=30
w_share=5
w_comment=4
w_collect=3
w_deep_view=2
w_view=1
user_port = pd.read_csv('userid_cateid_new.csv',index_col=[0])
user_id_list=list(user_port.index)
#print (user_id_list)
def cal_sim_list(user1):
    def cal_user_sim(user2):
        def distance(vector1,vector2):  
            d=0;  
            for a,b in zip(vector1,vector2):  
                d+=(a-b)**2;  
            return d**0.5;
        port_1=list(user_port.loc[user1])
        port_2=list(user_port.loc[user2])
        share_vec_1=[]
        for m in range (cate_num):
            share_vec_1.append(port_1[m])
        share_vec_2=[]
        for m in range (cate_num):
            share_vec_2.append(port_2[m])
        share_dist=distance(share_vec_1,share_vec_2)
        comment_vec_1=[]
        for m in range (cate_num):
            comment_vec_1.append(port_1[cate_num+m])
        comment_vec_2=[]
        for m in range (cate_num):
            comment_vec_2.append(port_2[cate_num+m])
        comment_dist=distance(comment_vec_1,comment_vec_2)
        collect_vec_1=[]
        for m in range (cate_num):
            collect_vec_1.append(port_1[2*cate_num+m])
        collect_vec_2=[]
        for m in range (cate_num):
            collect_vec_2.append(port_2[2*cate_num+m])
        collect_dist=distance(collect_vec_1,collect_vec_2)
        deep_view_vec_1=[]
        for m in range (cate_num):
            deep_view_vec_1.append(port_1[3*cate_num+m])
        deep_view_vec_2=[]
        for m in range (cate_num):
            deep_view_vec_2.append(port_2[3*cate_num+m])
        deep_view_dist=distance(deep_view_vec_1,deep_view_vec_2)
        view_vec_1=[]
        for m in range (cate_num):
            view_vec_1.append(port_1[4*cate_num+m])
        view_vec_2=[]
        for m in range (cate_num):
            view_vec_2.append(port_2[4*cate_num+m])
        view_dist=distance(view_vec_1,view_vec_2)
        user_sim=((w_share/(1+1000*share_dist))+(w_comment/(1+1000*comment_dist))+(w_collect/(1+1000*collect_dist))+(w_deep_view/(1+1000*deep_view_dist))+(w_view/(1+1000*view_dist)))/(w_share+w_comment+w_collect+w_deep_view+w_view)
        return user_sim
    user_sim_list=list(map(cal_user_sim,user_id_list))
    return user_sim_list
user_sim_mat=list(map(cal_sim_list,user_id_list))
user_sim_df = pd.DataFrame(user_sim_mat)
user_sim_df.index = user_id_list
user_sim_df.columns = user_id_list
user_sim_df.to_csv('user_sim_mat.csv',index=True)
print("输出完成")
logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' finished one part')
logfile1.close()