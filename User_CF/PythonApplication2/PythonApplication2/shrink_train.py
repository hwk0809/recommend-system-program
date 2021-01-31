import time
import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn import metrics
import copy
# _______________________________第一部分 生成userid_cateid_new.csv_________________________________________

# 计时
logfile1 = open("logfile_1.txt",'w')
logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' begin one part\n')

# 读入train
train = pd.read_csv('train.csv')

# 构建小型数据集
small_train=train.loc[0:16999];
small_user_list=list(set(small_train.user_id))
us_len = len(small_user_list)
for a in range(us_len):
    small_user_list[a] = re.compile(r'\n').sub('',small_user_list[a])
small_item_list=list(set(small_train.item_id))

pd.DataFrame(small_train).to_csv("small_train.csv",index=False)
print("small_train输出完成")
pd.DataFrame(small_user_list).to_csv("small_user_list.csv",index=False)
print("small_user_list输出完成")
pd.DataFrame(small_item_list).to_csv("small_item_list.csv",index=False)
print("small_item_list输出完成")