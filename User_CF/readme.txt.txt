Package          Version
---------------- ---------
backcall         0.2.0
certifi          2020.6.20
colorama         0.4.3
cycler           0.10.0
decorator        4.4.2
ipython          7.18.1
ipython-genutils 0.2.0
jedi             0.17.2
joblib           0.17.0
kiwisolver       1.2.0
matplotlib       3.3.2
networkx         2.5
numpy            1.19.2
pandas           1.1.4
parso            0.7.1
pickleshare      0.7.5
Pillow           7.2.0
pip              20.1.1
prompt-toolkit   3.0.7
Pygments         2.7.1
pyparsing        2.4.7
python-dateutil  2.8.1
pytz             2020.4
scikit-learn     0.23.2
scipy            1.5.4
setuptools       47.1.0
six              1.15.0
sklearn          0.0
threadpoolctl    2.1.0
traitlets        5.0.4
wcwidth          0.2.5


PythonApplication2（资讯推荐系统）:
1.运行shrink_train.py//读取train.csv，导出small_train、small_user_list、small_item_list三个小尺寸的csv文件，分别为训练集、用户列表、物品列表
2.运行collect_action.py//读取train.csv，导出action_freq.csv，得到用户行为和频次的数据，用于下一步分析不同行为应赋予的权重
3.运行item_col.py//读取small_train.csv，导出freq_num.csv，获得被阅读频次和资讯数量的关系，用于下一步分析热门资讯阈值
4.运行user_col.py//读取small_user_list和small_train两个csv文件，获得阅读篇数和用户数量的关系，用于下一步分析僵尸用户和深度用户阈值
5.运行user_portrait_small.py//读取small_user_list和small_train两个csv文件，导出hot_user、hot_item、userid_cateid_new三个csv文件，获得热门用户、热门新闻的数据，生成了用户画像矩阵
6.运行user_CF.py//读取userid_cateid_new.csv，导出user_sim_mat.csv，基于用户画像矩阵生成用户相似度矩阵
7.运行suggest.py//读取user_sim_mat、small_item_list、small_train三个csv文件，导出weight_mat、suggest_mat两个csv文件，获得物品推荐权重矩阵和推荐结果列表

PythonApplication4（电影评价系统）：
1.运行user_CF.py//读取zuoxiajiao_df2、youxiajiao_df2、zuoshangjiao_df2、youshangjiao_df2四个csv文件，导出user_sim_mat_2.csv，基于用户评分记录生成用户相似度矩阵
2.运行suggest.py//读取zuoxiajiao_df2、youxiajiao_df2、zuoshangjiao_df2、youshangjiao_df2、user_sim_mat_2五个csv文件，导出index_all_2.npy，获得物品预测评分列表，再生成MAE、precision-recall、DCG@N三种评价指标