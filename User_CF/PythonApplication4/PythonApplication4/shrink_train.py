import time
import pandas as pd
import numpy as np
import re
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn import metrics
import copy
# _______________________________第一部分 生成userid_cateid_new.csv_________________________________________


# 读入train
train = pd.read_csv('youxiajiao_df.csv',index_col=[0])

# 构建小型数据集
small_train=train.iloc[:400,:50]

pd.DataFrame(small_train).to_csv("youxiajiao_df2.csv")
print("输出完成")