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

# 设置阈值
user_threshold = 30
item_threshold = 30


# 读入用户ID，存入user_id_list列表
small_user_list = pd.read_csv('small_user_list.csv')
user_id_list = list(small_user_list.loc[:,'0'])
#print (user_id_list)

# 读入train
train = pd.read_csv('small_train.csv')

# 读入all_news_info,商品类别存入cate_id_list,商品名称存入item_id_list
cate_id = train.cate_id
cate_id_list = list(set(cate_id))
item_id = train.item_id
item_id_list = list(set(item_id))


# 构建活跃用户列表
def cal_hot_user(user):
    user_train = train.loc[train.user_id==user]
    user_items = list(set(user_train.item_id))
    user_num=len(user_items)
    if user_num>user_threshold:
        return user
    else:
        return '0'
hot_user_list = list(set(map(cal_hot_user,user_id_list)))
hot_user_list.remove('0')
pd.DataFrame(hot_user_list).to_csv("hot_user.csv",index=False)
print("hot_user输出完成")

# 构建热门新闻列表
item_train = list(train.item_id)
def cal_hot_item(item):
    click_rate = item_train.count(item)
    if click_rate>item_threshold:
        return item
    else:
        return '0'
hot_item_list = list(set(map(cal_hot_item,item_id_list)))
hot_item_list.remove('0')
pd.DataFrame(hot_item_list).to_csv("hot_item.csv",index=False)
print("hot_item输出完成")

# 构建DataFrame
def cal_eve_user(user):
    user_train_share = train.loc[(train.user_id==user) & (train.action_type=='share')]
    user_items_share = list(set(user_train_share.item_id))
    user_train_comment = train.loc[(train.user_id==user) & (train.action_type=='comment')]
    user_items_comment_raw = list(set(user_train_comment.item_id))
    user_items_comment=[]
    for m in user_items_comment_raw:
        if m not in user_items_share:
            user_items_comment.append(m)
    user_train_collect = train.loc[(train.user_id==user) & (train.action_type=='collect')]
    user_items_collect_raw = list(set(user_train_collect.item_id))
    user_items_collect=[]
    for m in user_items_collect_raw:
        if ((m not in user_items_share) &(m not in user_items_comment)):
            user_items_collect.append(m)
    user_train_deep_view = train.loc[(train.user_id==user) & (train.action_type=='deep_view')]
    user_items_deep_view_raw = list(set(user_train_deep_view.item_id))
    user_items_deep_view=[]
    for m in user_items_deep_view_raw:
        if ((m not in user_items_share) &(m not in user_items_comment)&(m not in user_items_collect)):
            user_items_deep_view.append(m)
    user_train_view = train.loc[(train.user_id==user) & (train.action_type=='view')]
    user_items_view_raw = list(set(user_train_view.item_id))
    user_items_view=[]
    for m in user_items_view_raw:
        if ((m not in user_items_share) &(m not in user_items_comment)&(m not in user_items_collect)&(m not in user_items_deep_view)):
            user_items_view.append(m)
    user_cold_items_share=[]
    user_hot_items_share=[]
    for m in user_items_share:
        if m not in hot_item_list:
            user_cold_items_share.append(m)
        else:
            user_hot_items_share.append(m)
    user_cold_items_comment=[]
    user_hot_items_comment=[]
    for m in user_items_comment:
        if m not in hot_item_list:
            user_cold_items_comment.append(m)
        elif m not in user_hot_items_share:
            user_hot_items_comment.append(m)
    user_cold_items_collect=[]
    user_hot_items_collect=[]
    for m in user_items_collect:
        if m not in hot_item_list:
            user_cold_items_collect.append(m)
        elif ((m not in user_hot_items_share)&(m not in user_hot_items_comment)):
            user_hot_items_collect.append(m)
    user_cold_items_deep_view=[]
    user_hot_items_deep_view=[]
    for m in user_items_deep_view:
        if m not in hot_item_list:
            user_cold_items_deep_view.append(m)
        elif ((m not in user_hot_items_share)&(m not in user_hot_items_comment)&(m not in user_hot_items_collect)):
            user_hot_items_deep_view.append(m)
    user_cold_items_view=[]
    user_hot_items_view=[]
    for m in user_items_view:
        if m not in hot_item_list:
            user_cold_items_view.append(m)
        elif ((m not in user_hot_items_share)&(m not in user_hot_items_comment)&(m not in user_hot_items_collect)&(m not in user_hot_items_deep_view)):
            user_hot_items_view.append(m)
    def find_itemcate(item):
        if item not in item_id_list:
            return
        onepart_item_cate = list(train.loc[train.item_id==item].cate_id)[0]
        return onepart_item_cate
    onepart_user_cate_list_view = list(map(find_itemcate,user_cold_items_view))
    onepart_user_cate_list_deep_view = list(map(find_itemcate,user_cold_items_deep_view))
    onepart_user_cate_list_comment = list(map(find_itemcate,user_cold_items_comment))
    onepart_user_cate_list_share = list(map(find_itemcate,user_cold_items_share))
    onepart_user_cate_list_collect = list(map(find_itemcate,user_cold_items_collect))
    auser_cate_num_list_view = list(map(lambda cate:onepart_user_cate_list_view.count(cate),cate_id_list))
    auser_cate_num_list_view.append(len(user_hot_items_view))
    auser_cate_num_list_deep_view = list(map(lambda cate:onepart_user_cate_list_deep_view.count(cate),cate_id_list))
    auser_cate_num_list_deep_view.append(len(user_hot_items_deep_view))
    auser_cate_num_list_comment = list(map(lambda cate:onepart_user_cate_list_comment.count(cate),cate_id_list))
    auser_cate_num_list_comment.append(len(user_hot_items_comment))
    auser_cate_num_list_share = list(map(lambda cate:onepart_user_cate_list_share.count(cate),cate_id_list))
    auser_cate_num_list_share.append(len(user_hot_items_share))
    auser_cate_num_list_collect = list(map(lambda cate:onepart_user_cate_list_collect.count(cate),cate_id_list))
    auser_cate_num_list_collect.append(len(user_hot_items_collect))
    auser_cate_num_list=auser_cate_num_list_share+auser_cate_num_list_comment+auser_cate_num_list_collect+auser_cate_num_list_deep_view+auser_cate_num_list_view
    return auser_cate_num_list
cold_user_list=[]
for m in user_id_list:
    if m not in hot_user_list:
        cold_user_list.append(m)
cold_user_cate = list(map(cal_eve_user,cold_user_list))
user_cate_df = pd.DataFrame(cold_user_cate)
user_cate_df.index = cold_user_list
cate_id_view=list(map(lambda cate:cate+'_view',cate_id_list))
cate_id_view.append('hot_view')
cate_id_deep_view=list(map(lambda cate:cate+'_deep_view',cate_id_list))
cate_id_deep_view.append('hot_deep_view')
cate_id_comment=list(map(lambda cate:cate+'_comment',cate_id_list))
cate_id_comment.append('hot_comment')
cate_id_share=list(map(lambda cate:cate+'_share',cate_id_list))
cate_id_share.append('hot_share')
cate_id_collect=list(map(lambda cate:cate+'_collect',cate_id_list))
cate_id_collect.append('hot_collect')
cate_id_new=cate_id_share+cate_id_comment+cate_id_collect+cate_id_deep_view+cate_id_view
user_cate_df.columns = cate_id_new
user_cate_df.to_csv('userid_cateid_new.csv',index=True)
print("输出完成")
logfile1.write(time.strftime('%Y-%m-%d %X',time.localtime(time.time()))+' finished one part')
logfile1.close()
