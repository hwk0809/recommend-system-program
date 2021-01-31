from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from pandas.plotting import radviz
import matplotlib
matplotlib.rc("font",family='YouYuan')
# 多维聚类数据可视化-画平行坐标图
# 输入：plot_data——原始数据（dataframe）
#      label—— 数据的聚类结果（list)
def plot_parallel_coordinates(plotdata,label):
    plotdata['label']=label
    print(plotdata)
    plt.figure(figsize=(100,50),dpi=20)
    parallel_coordinates(plotdata,'label')
    plt.show()
    dims = len(plotdata[0])
    print(dims)
    # fig, axes = plt.subplots(1, dims - 1, sharey=False)

def plot_radviz(plotdata,label):
    plotdata['label'] = label
    plt.figure('kmeans-radviz',figsize=(100,50))
    plt.title('radviz')
    radviz(plotdata, 'label')
    plt.show()

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    new_mt =pd.read_csv('new_mat_norm.csv')
    import Clustering_kmeans as km
    import Clustering_SOM as SO

    sommat = np.mat(new_mt.values)
    som = SO.SOM(sommat, (5, 5), 1, new_mt.shape[0])
    som.train()
    res = som.train_result()
    label =[]
    for i in range(len(res)):
        temp = res[i][0]
        label.append(temp)
    print(label)
    # k=km.clustering_kmeans_predict(new_mt,7,new_mt)
    # print(k)
    plot_radviz(new_mt, label)