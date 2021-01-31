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
train = pd.read_csv('small_train.csv')

# 读入all_news_info,商品名称存入item_id_list
item_id = train.item_id
item_id_list = list(set(item_id))

# 构建DataFrame
item_train = list(train.item_id)   
click_rate_list = list(map(lambda item:item_train.count(item),item_id))
freq_list = list(set(click_rate_list))
num_list = list(map(lambda freq:click_rate_list.count(freq),freq_list))
freq_num_df = pd.DataFrame(num_list)
freq_num_df.index = freq_list
freq_num_df.to_csv('freq_num.csv')
print("输出完成")
