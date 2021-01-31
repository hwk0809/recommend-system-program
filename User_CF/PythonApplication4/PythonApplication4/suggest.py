
import pandas as pd
import numpy as np
import re
import math
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics
from operator import itemgetter
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn import metrics
import copy
import time

# 计时
logfile1 = open("logfile_1.txt",'w')
logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' begin one part\n')



# 读入数据
LD= pd.read_csv('zuoxiajiao_df2.csv',index_col=[0])
RD= pd.read_csv('youxiajiao_df2.csv',index_col=[0])
LU= pd.read_csv('zuoshangjiao_df2.csv',index_col=[0])
RU= pd.read_csv('youshangjiao_df2.csv',index_col=[0])
user_sim_mat = pd.read_csv('user_sim_mat_2.csv',index_col=[0])

#预处理数据
Down=pd.concat([LD,RD],axis=1)
Left=pd.concat([LD,LU])
user_list=list(Left.index)
user_id_list=list(RU.index)
user_len=len(user_id_list)
item_id_list=list(RU.columns)
item_len=len(item_id_list)

def calc_rating(user_id,item_id,W):
    # 定义取top函数，返回索引值
    def Get_List_Max_Index(list_, n):
        N_large = pd.DataFrame({'score': list_}).sort_values(by='score', ascending=[False])
        return list(N_large.index)[:n]
    W_new=W.iloc[:,user_len:]

    # 生成权值矩阵
    def item_weight_list(user1):
        user_sim_list=list(W_new.loc[user1])
        sim_user_columns=Get_List_Max_Index(user_sim_list, 31)# 此处n值为相似用户数
        def item_weight(item):
            item_rank=list(RD.loc[:,item])
            weight=0
            for m in sim_user_columns:
                weight+=(user_sim_list[m])*(item_rank[m])
            return weight
        weight_list=list(map(item_weight,item_id_list))
        return weight_list
    weight_mat=list(map(item_weight_list,user_id_list))
    weight_mat_df = pd.DataFrame(weight_mat)
    weight_mat_df.index = user_id_list
    weight_mat_df.columns = item_id_list
    max_weight=0
    min_weight=10000
    for k in range(user_len):
        row=list(weight_mat_df.iloc[k])
        for l in range(item_len):
            if max_weight < row[l]:
                max_weight=row[l]
            if min_weight>row[l]:
                min_weight=row[l]
    rating=1+4*(weight_mat_df.loc[user_id].at[item_id]-min_weight)/(max_weight-min_weight)
    return rating

# 计算DCG指数
def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)

def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0
    ndcg = dcg / idcg
    return ndcg


def cal_4_index(U, T):
    T=np.array(T)
    N=2
    U_MAE=[]
    U_prec=[]
    U_recall=[]
    U_DCG=[]
    recommend_N = list(np.array(U).argsort()[::-1][0:10]) 
    trans_index = [sum(T>=2)+1,sum(T>=3)+1,sum(T>=4)+1,sum(T>=5)+1,1]
    while N<=10:
        U_MAE.append(metrics.mean_absolute_error(T, U))
        U_prec.append(sum(T[recommend_N[:N]]>=4)/min(N,len(T)))
        U_recall.append(sum(T[recommend_N[:N]]>=4)/sum(T>=4))
        true_rank = [trans_index[T[i]-1] for i in recommend_N[:N]]
        U_DCG.append(getNDCG(true_rank, list(range(1, min(len(T),N)+1))))
        N = N+1
    return [U_MAE, U_prec, U_recall, U_DCG]

# 生成ratings预测值文件
test_userid = RU.index.values.tolist()
expected_itemid = RU.columns.values.tolist()
predict_rating = {}
all_pred_label = []
all_true_label = []
index_all = []
for userid in test_userid:
    sub_pred_label = []
    sub_true_label=[]
    print(1)
    for itemid in expected_itemid:
        a=RU.iloc[test_userid.index(userid)][itemid]
        if (a==0):
            continue
        b = calc_rating(userid, itemid, user_sim_mat)
        sub_pred_label.append(b)
        sub_true_label.append(a)
        all_pred_label.append(b)
        all_true_label.append(a)
    if (len(sub_true_label)==0):
        continue
    index_all.append(cal_4_index(sub_pred_label, sub_true_label))
np.save('index_all_2.npy',index_all)
print('输出完成')
logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' finished one part')
logfile1.close()