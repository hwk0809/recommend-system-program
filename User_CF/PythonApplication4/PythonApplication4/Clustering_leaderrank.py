# -*- coding: UTF-8 -*-

"""
Created on 18-3-5

@summary: LeaderRank 节点排序算法

@author: hwk
"""
import networkx as nx
import numpy as np


def leaderrank(graph):
    """
    节点排序
    :param graph:复杂网络图Graph
    :return: 返回节点排序值
    """
    # 节点个数
    num_nodes = graph.number_of_nodes()
    # 节点
    nodes = graph.nodes()
    # 在网络中增加节点g并且与所有节点进行连接
    graph.add_node(0)
    for node in nodes:
        graph.add_edge(0, node)
        graph.add_edge(node, 0)
    # LR值初始化
    LR = dict.fromkeys(nodes, 1.0)
    LR[0] = 0.0
    # 迭代从而满足停止条件
    while True:
        tempLR = {}
        for node1 in graph.nodes():
            s = 0.0
            for node2 in graph.nodes():
                if graph.has_edge(node2, node1):
                    s += 1.0 / graph.out_degree([node2])[node2] * LR[node2]
            tempLR[node1] = s
        # 终止条件:LR值不在变化
        error = 0.0
        for n in tempLR.keys():
            error += abs(tempLR[n] - LR[n])
        if error == 0.0:
            break
        LR = tempLR
    # 节点g的LR值平均分给其它的N个节点并且删除节点
    avg = LR[0] / num_nodes
    LR.pop(0)
    for k in LR.keys():
        LR[k] += avg

    return LR


if __name__ == "__main__":
    import pandas as pd

    # demo1
    # Matrix = np.mat([[0,1,0,0,1,0],[0,0,1,0,0,0],[1,0,0,1,1,0],[0,1,0,0,0,1],[0,1,0,1,0,1],[1,0,0,0,0,0]])
    # graph = nx.from_numpy_matrix(Matrix,create_using=nx.DiGraph)
    # H = nx.relabel_nodes(graph,dict(enumerate(['user1','user2','user3','user4','user5','user6'])),copy=False)

    # demo2
    Matrix = np.mat([[0,1,0,1],[0,0,1,1],[0,1,0,1],[0,1,1,0]])
    graph = nx.from_numpy_matrix(Matrix,create_using=nx.DiGraph)
    H = nx.relabel_nodes(graph,dict(enumerate(['user1','user2','user3','user4'])),copy=False)
    print(leaderrank(graph))
    # print(sorted(leaderrank(graph).items(), key=lambda item: item[1]))
    # import matplotlib.pyplot as plt
    #
    # plt.figure()
    # nx.draw(graph, with_labels=True, font_color='#000000', node_color='r', font_size=8, node_size=20)
    # plt.show()

    # print sorted(leaderrank(graph).items(), key=lambda item: item[1])
