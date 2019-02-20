#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: T


# 导入数据集
from sklearn.datasets import make_blobs
# 导入sklearn knn算法
from sklearn.neighbors import KNeighborsClassifier
# 导入数值计算库
import numpy as np 
# 导入画图库
import matplotlib.pyplot as plt
# 指定默认字体 解决画图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

'''
K近邻算法 思想：不知其人视其友
'''
'''
K近邻算法（knn)实现步骤：
	1.计算当前点与已知类别数据集中的点的距离
	2.距离递增排序，选出距离最小的k个点
	3.确定前k个点类别出现的频率
	4.将频率最高的类别最为当前点的预测分类
'''

def knn(inX, dataSet, label, k):
	'''
		inX(np.array): 需要预测的样本点
		dataSet(np.array): 已知类别的数据集 
		label(list): 数据集的样本类型
		K(int): 最近邻数目
	'''
	# 预测样本点的维数
	p_row, p_col = inX.shape
	# 已知样本点的维数
	s_row, s_col = dataSet.shape

	# 定义预测结果集
	list_res = []
	# 遍历预测样本点
	for i in range(p_row):
		# 遍历已知样本点并计算欧式距离
		list_dist = [distance(inX[i, ], dataSet[j, ]) for j in range(s_row)]
		# 获取按升序排序的索引
		mink_index = np.argsort(list_dist)
		# 统计前K个点出现的类别个数
		classCount = {}
		for c in range(k):
			voteClass = label[mink_index[c]]
			classCount[voteClass] = classCount.get(voteClass, 0) + 1
		# 获取个数最大的类别
		maxClass = max(classCount, key=classCount.get)
		# 添加到预测结果集
		list_res.append(maxClass)
	
	showScatter(dataSet, label, inX, list_res, title='knn预测样本结果')

	return list_res

# 计算向量欧式距离
def distance(vecX, vecY):
	return np.linalg.norm(vecX - vecY)

# sklearn 构造分类数据集
def createData():
	'''
	n_samples: 样本总数
	n_features： 样本特征数
	centers：样本类别数
	cluster_std：样本类内方差
	'''
	dataSet, label = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=[1.4, 1.5, 2.0])

	# 数据划分
	n_row, n_col = dataSet.shape
	train_data = dataSet[:-10]
	test_data = dataSet[n_row-10:]
	train_label = label[:-10]
	test_label = label[n_row-10:]

	# 画散点图 
	showScatter(train_data, train_label, test_data, 'b', title='原始数据散点图')

	return train_data, train_label, test_data, test_label

# 画散点图
def showScatter(train_data, train_color, test_data, test_color, title):
	'''
	o 表示训练样本
	v 表示测试样本
	'''
	plt.title(title)
	plt.scatter(train_data[:, 0], train_data[:, 1], c=train_color, marker='o')
	plt.scatter(test_data[:, 0], test_data[:, 1], c=test_color, marker='v')
	plt.show()

def sklearn_knn(inX, dataSet, label, k):
	# 定义knn分类器对象
	knn = KNeighborsClassifier()
	# 训练分类器
	knn.fit(dataSet, label)
	# 利用分类器进行样本预测
	pre_res = knn.predict(inX)

	# 画散点图
	showScatter(dataSet, label, inX, pre_res, title='sklearn-knn预测样本结果')

	return pre_res

def main():
	# 构造数据集
	train_data, train_label, test_data, test_label  = createData()
	# knn算法预测
	knn(test_data, train_data, train_label, k=10)
	# sklearn-knn算法预测
	sklearn_knn(test_data, train_data, train_label, k=10)


if __name__ == '__main__':
	main()