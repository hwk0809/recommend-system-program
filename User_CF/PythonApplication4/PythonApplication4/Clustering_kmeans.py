#  Kmeans聚类函数
#  输入：k——聚类数目
#       df——聚类训练集   输入用户商品类别矩阵（列名称为商品类别，每行代表一个用户）
#       df——聚类预测集
#  输出：预测结果,以列表形式输出
from sklearn.cluster import KMeans
import numpy as np

def clustering_kmeans(df, k, init_input=None):
    # 如果没输入参数
    if init_input is None:
        init_input = 'k-means++'
        n=10
    else:
        init_input =np.array(init_input)
        n=1

    # kmeans聚类
    km = KMeans(n_clusters=k,n_init=n,init=init_input)
    K = km.fit(df.values)
    train_label = km.labels_  # 训练数据

    # 转换成字典格式
    classify = {}
    for i, win in enumerate(train_label):
        if not classify.get(win):
            classify.setdefault(win, [i])
        else:
            classify[win].append(i)

    return classify


def clustering_kmeans_predict(df,k,df_test,init_input=None):
    if init_input is None:
        init_input = 'k-means++'
        n = 10
    else:
        init_input = np.array(init_input)
        n = 1
    km = KMeans(n_clusters=k, n_init=n, init=init_input)
    K = km.fit(df.values)
    train_label = km.labels_  # 训练数据
    # test_label = km.predict(df_test.values)  # 预测数据
    # overduetimes_predicted = KMeans(n_clusters=3, n_init=1, init=np.array([[0], [5], [10]])).fit(X).predict(
    #     X)  # init=np.array,选择聚类中心
    return train_label


# from sklearn.cluster import k_means_
# from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
# from sklearn.preprocessing import StandardScaler
#
#
# def create_cluster(sparse_data, nclust=10):
#     # Manually override euclidean
#     def euc_dist(X, Y=None, Y_norm_squared=None, squared=False):
#         # return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
#         return cosine_similarity(X, Y)
#
#     k_means_.euclidean_distances = euc_dist
#
#     scaler = StandardScaler(with_mean=False)
#     sparse_data = scaler.fit_transform(sparse_data)
#     kmeans = k_means_.KMeans(n_clusters=nclust, n_jobs=20, random_state=3425)
#     _ = kmeans.fit(sparse_data)
#     return kmeans.labels_



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from sklearn.datasets import make_blobs
    import pandas as pd
    # X为样本特征，Y为样本簇类别，共1000个样本，每个样本2个特征，对应x和y轴，共4个簇，
    # 簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
    X, y = make_blobs(n_samples=1000, n_features=2, centers=[[10, 10], [20, 20], [30, 30], [40, 40]],
                      cluster_std=[4, 2, 2, 2], random_state=9)

    plt.scatter(X[:, 0], X[:, 1], marker='o')  # 假设暂不知道y类别，不设置c=y，使用kmeans聚类
    plt.show()
    df=pd.DataFrame(X,y,columns=['x','y'])
    print(df)
    # np.r_[d, [[5, 6]]]
    k=3
    m = np.array([[10,10],[20,20],[30,30]])
    print(m)
    labels=clustering_kmeans(df, k,init_input=m)
    plt.scatter(df['x'], df['y'], c=labels)
    plt.show()








