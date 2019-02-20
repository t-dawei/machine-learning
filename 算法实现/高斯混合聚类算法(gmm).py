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
# 导入sklearn GMM算法
from sklearn import mixture
# 导入作图库
import matplotlib.pyplot as plt
# 指定默认字体 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['font.family'] = 'sans-serif'
# 忽略np warning
# np.seterr(divide='ignore', invalid='ignore')
# 导入NMI
from NMI import myNMI, sklearnNMI
# 导入统计库
import collections
# 导入 xie-beni
from xie_beni import xieBeni


# 高斯分布的概率密度函数 x为单个样本数据 mu为均值 cov为协方差
def gsProb(x, mu, cov):
	n = x.shape[1]
	expOn = float(-0.5 * (x - mu) * (cov.I) * ((x - mu).T))
	# np.linalg.det 计算矩阵的行列式
	divBy = pow(2 * np.pi, n / 2) * pow(np.linalg.det(cov), 0.5)
	return pow(np.e, expOn) / divBy  


 # EM 算法 聚类个数为3 迭代次数为100
def EM(np_data, maxIter=100):
	# 获取数据的样本数和维度
	rows, cols = np_data.shape
	# 1.初始化各高斯混合成分参数pai
	pai = [1/3, 1/3, 1/3]
	# 设置均值 使用kmean的中心样本
	# mu = [np_data[6, :], np_data[22, :], np_data[30, :]]
	mu = [np.array([2.94805141, 2.84585132]), np.array([7.16504475, 7.12121176]), np.array([4.92859254, 4.93144926])]
	# 初始化协方差cov
	cov = [np.mat([[0.1, 0], [0, 0.1]]) for x in range(3)]
	# 初始化后验概率
	gamma = np.mat(np.zeros((rows, 3)))

	# 迭代次数
	for i in range(maxIter):

		# 对每个样本的后验概率求和
		for j in range(rows):		
			sumAlphaMulP = 0
			for k in range(3):
				# 计算每个样本混合成分生成的后验概率
				gamma[j, k] = pai[k] * gsProb(np.mat(np_data[j, :]), mu[k], cov[k])
				sumAlphaMulP += gamma[j, k]
			# 求比例
			for k in range(3):
				gamma[j, k] /= sumAlphaMulP
		sumGamma = np.sum(gamma, axis=0)

		for k in range(3):
			mu[k] = np.mat(np.zeros((1, cols)))
			cov[k] = np.mat(np.zeros((cols, cols)))

			# 计算新均值
			for j in range(rows):
				mu[k] += gamma[j, k] * np_data[j, :]
			mu[k] /= sumGamma[0, k]

			# 计算新的协方差矩阵
			for j in range(rows):
				cov[k] += gamma[j, k] * (np_data[j, :] - mu[k]).T *(np_data[j, :] - mu[k])
			cov[k] /= sumGamma[0, k]

			# 计算新混合系数
			pai[k] = sumGamma[0, k] / rows
	
	# 打印混合高斯模型参数
	print('高斯混合模型比例系数：\n{}'.format(pai))
	print('高斯混合模型均值向量：\n{}'.format(mu))
	print('高斯混合模型协方差矩阵：\n{}'.format(cov))

	return gamma, mu



def my_Gmm(np_data):

	# 获取实验数据的样本数和维数
	rows, cols = np_data.shape

	# 调用 EM 算法 返回后验概率
	gamma, centroids = EM(np_data)
	# 初始化 preCluster rows * 2 第一列表示预测聚类结果 第二列表示概率值
	preCluster = np.mat(np.zeros((rows, 2)))
	# 对每个样本进行预测
	for i in range(rows):
		# amx返回矩阵最大值，argmax返回矩阵最大值所在下标
		preCluster[i, :] = np.argmax(gamma[i, :]), np.amax(gamma[i, :])

	# 可视化分类结果
	res_predict = [int(x[0]) for x in preCluster[:,0].tolist()]
	# print(res_predict)
	stand_predict = standardization(res_predict)
	graph(data=np_data, color=res_predict, marker='o', title='myGMM')

	return centroids, stand_predict, gamma



# 调用sklearn 的高斯混合聚类算法查看聚类效果 用于结果对比
def sklearn_Gmm(data, components=3, iter=100, cov_type="full"):

	# 调用算法 3类 最大迭代100词
	gm = mixture.GaussianMixture(n_components=components, max_iter=iter, covariance_type=cov_type)
	# 训练模型
	gm.fit(data)
	# 预测分类
	pred_labels = gm.predict(data)

	# 分类结果可视化
	graph(data=data, color=pred_labels, marker='o', title='sklearnGMM')
	return pred_labels


# 画散点图
def graph(data, color, marker, title, xlabel=None, ylabel=None):
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


# 读取 excel 数据
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

	# 删除含有空值的数据
	pd_data = pd_data.dropna()

	# 查看原始数据
	# print(pd_data)
	# 转化为np对象
	np_data = np.array(pd_data)

	datasets = np_data[:,0:2]
	labels = np_data[:, 2]

	new_lables = []
	for i in labels:
		new_lables.append(int(i))

	# 原始数据图
	graph(data=np_data, color=new_lables, marker='o', title='原始数据图', xlabel='X', ylabel='Y')

	return np_data[:,0:2], new_lables

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

def main():

	# 读取实验数据
	np_data, labels = readExcel()

	# 调用sklearn 测试实验数据聚类结果
	pred_labels = sklearn_Gmm(np_data)

	# 调用高斯混合算法
	centroids, stand_predict, gamma = my_Gmm(np_data)
	
	# sklearnNMI(labels, stand_predict)
	myNMI(labels, stand_predict)
	
	xieBeni(gamma.tolist(), np.array(centroids), np_data)


if __name__ == '__main__':
	main()