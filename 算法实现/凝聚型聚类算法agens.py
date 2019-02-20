#!/usr/bin/python
# -*- coding: utf-8 -*-

# 导入读取xl库
import xlrd
# 导入索引矩阵计算库
import pandas as pd
# 导入数学计算库
import numpy as np
# 导入sklearn 数据集
from sklearn import datasets
# 导入sklearn 均值算法
from sklearn.cluster import KMeans
# 导入作图库
import matplotlib.pyplot as plt
# 指定默认字体 解决画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
# 导入统计库
import collections


# 读取 excel 数据 老师给的数据集 实验用
def readExcel():
    path = r'chap_7_大作业数据集.xlsx'
    # 打开文件
    wb = xlrd.open_workbook(path)
    # 查看所有的sheet
    # print(wb.sheet_names())
    # 取第一个 sheet 的名称
    sheet_name = wb.sheet_names()[0]
    # 通过名称获取sheet对象
    sheet = wb.sheet_by_name(sheet_name)
    # 查看sheet的行数和列数
    # print(sheet.nrows, sheet.ncols)

    # 这里用pandas读取生成矩阵 也可以用xlrd读取
    pd_data = pd.read_excel(io=path, sheet_name=sheet_name, header=None)
    # 查看原始数据
    # print(pd_data)
    # 转化为np对象
    np_data = np.array(pd_data)
    labels = np_data[:,2]

    new_lables = []
    for i in labels:
        new_lables.append(int(i))

    # 原始数据图
    # graph(data=np_data, color=new_lables, marker='o', title='原始数据图', xlabel='X', ylabel='Y')

    return np_data[:,0:2], new_lables

# 画散点图
def graph(data, color, marker, title, xlabel=None, ylabel=None):
    # plt.ion()
    # 设置标题
    plt.title(title)
    plt.scatter(data[:, 0], data[:, 1], c=color, marker=marker)
    if xlabel: 
        plt.xlabel(xlabel)  
    if ylabel:
        plt.ylabel(ylabel)
    # 设置图标 loc=2表示左上角
    # plt.legend(loc=2) 
    plt.show()

import math
# 计算欧式距离
def distance(a, b):
	return math.sqrt(math.pow(a[0]-b[0], 2) + math.pow(a[1]-b[1], 2))


# Average Linkage算法
def dist_avg(Ci, Cj):
	return sum(distance(i, j) for i in Ci for j in Cj)/(len(Ci)*len(Cj))

def findMin(M):
	min_ = float('inf')
	x = y = 0
	for i in range(len(M)):
		for j in range(i+1):
			if i != j and M[i][j] < min_:
				min_, x, y = M[i][j], i, j
	return x, y, min_


# 预测结果标准化 观察结果 按频率
def standardization(res):

    obj = collections.Counter(res)
    # print(obj)
    list_order = []
    for sz in obj.most_common():
        # print(key)
        list_order.append(sz[0])

    new_res = []
    for r in res:
        if r == list_order[0]:
            new_res.append(1)
        elif r == list_order[1]:
            new_res.append(2)
        elif r == list_order[2]:
            new_res.append(3)
        else:
            print('数据有误')
    return new_res

def myAgens(np_data, k=3):
	n, m = np_data.shape
	# 初始簇
	C = [[(i[0], i[1])] for i in np_data]
	# 初始化距离矩阵
	M = []
	for i in range(len(C)):
		Mi = [dist_avg(C[i], C[j]) for j in range(i+1)]
		M.append(Mi)

	while len(C) > k:
		x, y, min_ = findMin(M)
		# 更新C
		C[x].extend(C[y])
		C.remove(C[y])
		# 更新M
		M.clear()
		for i in range(len(C)):
			Mi = [dist_avg(C[i], C[j]) for j in range(i+1)]
			M.append(Mi)
		print(len(C))

	# 转化成预测结果
	res = []
	for i in np_data:
		if (i[0], i[1]) in C[0]:
			res.append(1)
		elif (i[0], i[1]) in C[1]:
			res.append(2)
		elif (i[0], i[1]) in C[2]:
			res.append(3)
		else:
			print('数据有误')

	stand_predict = standardization(res)
	graph(data=np_data, color=stand_predict, marker='o', title='凝聚层次聚类', xlabel='X', ylabel='Y')

	# 聚类中心
	clucenters = np.zeros((k, 2))
	for i in range(len(C)):
		newClust = np.array(C[i])
		clucenters[i, :] = np.mean(newClust, axis=0)

	# 构造u -- xb指标使用
	u = np.zeros((n, k), dtype=np.int)
	for index, item in enumerate(stand_predict):
		u[index][item-1] = 1

	print(clucenters)
	print(u)
	return clucenters, stand_predict, u


# 主函数
def main():
    np_data, labels = readExcel()
    
    myAgens(np_data, 3)


if __name__ == '__main__':
	main()