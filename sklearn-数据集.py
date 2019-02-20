#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: T

from sklearn import datasets


# make_blobs 模块
dataSet, label = datasets.make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=[1.4, 1.5, 2.0])
'''
n_samples: 样本总数
n_features： 样本特征数
centers：样本类别数
cluster_std：样本类内方差
'''

# load 模块
iris = datasets.load_iris()