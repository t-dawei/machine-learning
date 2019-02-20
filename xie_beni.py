#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import numpy as np 

# 这是网上参考代码： https://blog.csdn.net/lanse_zhicheng/article/details/78999788
def xie_beni(membership_mat,center,data):
    sum_cluster_distance=0
    min_cluster_center_distance=inf
    for i in range(k):
        for j in range(n):
            sum_cluster_distance=sum_cluster_distance + membership_mat[j][i]** 2 * sum(power(data[j,:]- center[i,:],2))#计算类一致性
    for i in range(k-1):
        for j in range(i+1,k):
            cluster_center_distance=sum(power(center[i,:]-center[j,:],2))#计算类间距离
            if cluster_center_distance<min_cluster_center_distance:
                min_cluster_center_distance=cluster_center_distance
    return sum_cluster_distance/(n*min_cluster_center_distance)


# 实验使用代码 
# 结果越小 聚类效果越好 n=2
# 公式：类内紧凑度 / 类间分离度
# u[j][i] 表示第j个样本属于第i类的隶属度
def xieBeni(u, clucenters, np_data):
    # 获取样本个数
    n = np_data.shape[0]
    # 获取聚类个数
    k = clucenters.shape[0]
    # 定义类内紧凑度
    sum_cluster_distance = 0

    for i in range(k):
        for j in range(n):
            # 计算类一致性
            sum_cluster_distance += math.pow(u[j][i], 2) * np.sum(np.power(np_data[j,:] - clucenters[i,:], 2))
    # print(sum_cluster_distance)

    # 定义类间分离度
    min_center_distance = float('inf')
    for i in range(k-1):
        for j in range(i+1, k):
            # 计算类间距离
            center_distance = np.sum(np.power(clucenters[i,:] - clucenters[j,:], 2))
            if center_distance < min_center_distance:
                min_center_distance = center_distance
    # print(min_center_distance)
    res = sum_cluster_distance / (n * min_center_distance)
    print(res)
    return res