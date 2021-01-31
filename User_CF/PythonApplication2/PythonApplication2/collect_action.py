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

# 读入train
train = pd.read_csv('train.csv')
action_list=list(train.action_type)
action_type_list=list(set(train.action_type))
action_freq = list(map(lambda action:action_list.count(action),action_type_list))
action_freq_df = pd.DataFrame(action_freq)
action_freq_df.index = action_type_list
action_freq_df.to_csv('action_freq.csv')
print("输出完成")
