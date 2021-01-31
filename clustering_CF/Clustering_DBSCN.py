#  Kmeans聚类函数
#  输入：k——聚类数目
#       df——聚类训练集   输入用户商品类别矩阵（列名称为商品类别，每行代表一个用户）
#       df——聚类预测集
#       DistanceFlat   0 -欧氏距离，1-余弦距离 不输入时默认为欧氏距离
#       r———样本邻域
#       min_number——集群的最小个数
#  输出：预测结果,以列表形式输出
from sklearn.cluster import DBSCAN
import numpy as np
import Clusering_evaluate as EV

def clustering_DBSCAN(df,DistanceFlag=1,r=0.3,min_number=30):

    #choose Distance
    d = 'euclidean'
    if DistanceFlag == 0:
        d = 'euclidean'
    elif DistanceFlag == 1:
        d = 'cosine'

    # Compute DBSCAN
    db = DBSCAN(eps=r, min_samples=min_number, metric=d).fit(df.values)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
    print('噪声比:', format(raito, '.2%'))
    print('Estimated number of clusters: %d' % n_clusters_)
    # EV.print_silhouette(df.values,labels)

    # 转换成字典格式
    classify = {}
    for i, win in enumerate(labels):
        if not classify.get(win):
            classify.setdefault(win, [i])
        else:
            classify[win].append(i)

    return classify

def clustering_DBSCAN_predict(df,df_test,DistanceFlag=0):

    #choose Distanc
    if DistanceFlag == 0:
        d = 'euclidean'
    elif DistanceFlag == 1:
        d = 'cosine'

    # Compute DBSCAN
    db = DBSCAN(eps=0.03, min_samples=8, metric=d).fit(df.values)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
    print('噪声比:', format(raito, '.2%'))
    print('Estimated number of clusters: %d' % n_clusters_)
    # EV.print_silhouette(df.values,labels)
    test_label=db.fit_predict(df_test.values)
    return test_label

if __name__ == "__main__":
    # 用户-商品类别矩阵
    import pandas as pd
    df = pd.read_csv('userid_cateid_new.csv')
    test_label = clustering_DBSCAN(df, df,1)
    print(test_label)







