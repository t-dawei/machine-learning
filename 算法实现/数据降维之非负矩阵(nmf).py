#!/usr/bin/env python
#-*- coding:utf-8 -*-

# 数值计算库
import numpy as np
# 科学计算库
import pandas as pd
# 数据预处理库-特征数据标准化
from sklearn.preprocessing import StandardScaler
# sklearn算法库
from sklearn.decomposition import NMF
# 数据集
from sklearn import datasets
# numpy 随机数
from numpy.random import RandomState
rng = RandomState(0)
# 图片加载库
from PIL import Image
# 画图库
import matplotlib.pyplot as plt
# 读取mat文件
import scipy.io as sio

def readData():
	# 加载数据集
	# digits = datasets.load_digits()
	# iris = datasets.load_iris()
	# load_data = sio.loadmat('olivettifaces.mat')
	# faces = pd.DataFrame(pd.read_csv('olivettifaces.csv'))
	
	# 加载face_images.mat数据集
	# load_data = sio.loadmat('face_images.mat')
	# image = load_data['images']
	# pd_data = pd.DataFrame(image)
	# pd_data = pd_data.T
	# print(pd_data.shape)
	# print(pd_data)

	faces = datasets.fetch_olivetti_faces(shuffle=True, random_state=rng)

	# 创建pandas对象
	pd_data = pd.DataFrame(faces.data)
	print(pd_data.shape)

	# 删除含有空值的数据
	pd_data = pd_data.dropna()

	# 查看前10行特征数据
	# print(pd_data.head(10))

	# 查看特征名称
	# print(pd_data.columns)

	# 单独提取特征数据 生成numpy对象 用于计算
	np_data = np.array(pd_data)

	# np_data = np_data[:10]
	# print(np_data)
	# face_titles = ["face %d" % i for i in range(np_data.shape[0])]
	# plot_gallery(np_data, face_titles, 64, 64)


	# 查看特征数据的维度
	print('原始数据的维度：{}'.format(np_data.shape))
	#　查看详细特征数字
	print('原始数据：\n{}'.format(np_data))
	#　如果不同维度的数据间量级相差较大，就需要预先对数据进行标准化处理
	# np_data = standardData(np_data)
	# print(np_data)
	return np_data


# sklearn自带NMF降维函数 feat表示保留的特征维数
def sklearn_NMF(np_data, feat=16):
	# 创建NMF对象  W, H 初始化方式
	nmf = NMF(n_components=feat, init='nndsvda', tol=5e-3)
	# 使用NMF对特征数据进行降维 得到权重矩阵
	res_data = nmf.fit_transform(np_data)

	# 查看权重矩阵的维度 
	print('sklearn-NMF 数据降维后权重矩阵维度：{}'.format(res_data.shape))
	# 查看降维后权重矩阵数据
	print('sklearn-NMF 数据降维后权重矩阵数据：\n{}'.format(res_data))

	# 查看数据降维后特征矩阵数据
	components_ = nmf.components_
	print('sklearn-nmf模型的特征矩阵维度：{}'.format(components_.shape))
	print('sklearn-nmf模型的特征矩阵数据：\n{}'.format(components_))

	# 重构数据 
	reconMat = np.dot(res_data, components_)
	print('sklearn-nmf 数据重构后维度：{}'.format(reconMat.shape))
	print('sklearn-nmf 数据重构后数据：\n{}'.format(reconMat))
	return res_data, reconMat 



# np_data 矩阵数据， feat保留特征
def my_NMF(np_data, feat=16, iter_=50, error=0.0):
	row, col = np_data.shape
	# 初始化权重矩阵
	W = np.mat(np.random.random((row, feat)))
	# 初始化特征矩阵
	H = np.mat(np.random.random((feat, col)))
	# 迭代次数
	for k in range(iter_):
		print('当前迭代次数为{}'.format(k+1))
		# W H 都为matrix对象 * 表示为矩阵乘积
		WH = W * H
		# 计算误差
		E = np_data - WH
		cost = 0.0
		for i in range(row):
			for j in range(col):
				cost += E[i,j] * E[i,j]
		# 显示其过程
		if k%10 == 0:
			print(cost)
		# 结束迭代
		if cost == error:
			break
		# 权重矩阵的转置矩阵 * data matrix
		HN = W.T * np_data
		# 权重矩阵d的转置矩阵 * 权重矩阵 * 特征矩阵
		HD = W.T * W * H
		# 更新特征矩阵
		# H = np.mat(np.array(H) * np.array(HN) / np.array(HD))
		for f in range(feat):
			for c in range(col):
				if HD[f,c] != 0:
					H[f,c] = H[f,c] * HN[f,c] / HD[f,c] 

		# #data matrix * 特征矩阵转置矩阵
		WN = np_data * H.T
		# 权重矩阵 * 特征矩阵 * 特征矩阵转置
		WD = W * H * H.T
		# 更新权重矩阵
		# W = np.mat(np.array(W) * np.array(WN) / np.array(WD))
		for r in range(row):
			for f in range(feat):
				if WD[r,f] != 0:
					W[r,f] = W[r,f] * WN[r,f] / WD[r,f]

	print('权重矩阵的维度：\n{}'.format(W.shape))
	print('权重矩阵数据：\n{}'.format(W))
	print('特征矩阵的维度：\n{}'.format(H.shape))
	print('特征矩阵数据：\n{}'.format(H))

	# 重构数据
	WH = W * H
	print('重构矩阵的维度：\n{}'.format(WH.shape))
	print('重构矩阵数据：\n{}'.format(WH))
	return W, H, WH


# 特征数据标准化
def standardData(data):
	# 创建StandardScaler对象
	sc = StandardScaler()
	sc_data = sc.fit_transform(data)
	return sc_data

def plot_gallery(images, titles, h, w, n_row=2, n_col=5):
	plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
	plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
	for i in range(n_row * n_col):
		data = images[i:i+1]
		plt.subplot(n_row, n_col, i + 1)
		plt.imshow(data.reshape((h, w)), cmap=plt.cm.gray)
		# plt.title(titles[i], size=12)
		plt.xticks(())
		plt.yticks(())
	plt.savefig('tt.png')
	plt.show()


# 主函数
def main():
	np_data = readData()
	res_data, reconMat = sklearn_NMF(np_data)
	face_titles = ["face %d" % i for i in range(np_data.shape[0])]
	plot_gallery(reconMat, face_titles, 64, 64)

	# W, H, WH = my_NMF(np_data)
	# face_titles = ["face %d" % i for i in range(np_data.shape[0])]
	# plot_gallery(WH, face_titles, 64, 64)


if __name__ == '__main__':
	main()

