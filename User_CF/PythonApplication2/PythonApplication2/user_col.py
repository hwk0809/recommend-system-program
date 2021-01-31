
import time
import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn import metrics
import copy

# 读入用户ID，存入user_id_list列表
small_user_list = pd.read_csv('small_user_list.csv')
user_id_list = list(small_user_list.loc[:,'0'])

# 读入train
train = pd.read_csv('small_train.csv')

# 读入all_news_info,商品类别存入cate_id_list
cate_id = train.cate_id
cate_id_list = list(set(cate_id))

# 构建DataFrame

def cal_eve_user(user):
    user_train = train.loc[train.user_id==user]
    user_items = list(set(user_train.item_id))
    user_num=len(user_items)
    return user_num
user_num_list = list(map(cal_eve_user,user_id_list))
num_list = list(set(user_num_list))
mem_list = list(map(lambda num:user_num_list.count(num),num_list))
num_mem_df = pd.DataFrame(mem_list)
num_mem_df.index = num_list
num_mem_df.to_csv('num_mem.csv')
print("输出完成")