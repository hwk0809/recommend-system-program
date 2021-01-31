# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# 轮廓系数
def print_silhouette(X,labels):
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, labels))

def cal_silhouette(X,labels):
    return metrics.silhouette_score(X, labels)
