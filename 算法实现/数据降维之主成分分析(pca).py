#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @auther: T

# 数值计算库
import numpy as np
# 科学计算库
import pandas as pd
# 数据预处理库-特征数据标准化
from sklearn.preprocessing import StandardScaler
# PCA算法库
from sklearn.decomposition import PCA
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

	# 加载face_images.mat数据集
	# load_data = sio.loadmat('face_images.mat')
	# image = load_data['images']
	# pd_data = pd.DataFrame(image)
	# pd_data = pd_data.T
	# print(pd_data.shape)
	# print(pd_data)

	# 加载
	# load_data = sio.loadmat('AR.mat')
	# image = load_data['AR']
	# pd_data = pd.DataFrame(image)
	
	# 加载 sklearn 提供的可降维数据集
	# digits = datasets.load_digits()
	# iris = datasets.load_iris()
	
	# 加载sklearn 人脸数据集
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
	face_titles = ["face %d" % i for i in range(np_data.shape[0])]
	plot_gallery(np_data, face_titles, 64, 64)


	# 查看特征数据的维度
	print('原始数据的维度：{}'.format(np_data.shape))
	#　查看详细特征数字
	print('原始数据：\n{}'.format(np_data))
	#　如果不同维度的数据间量级相差较大，就需要预先对数据进行标准化处理
	# np_data = standardData(np_data)
	# print(np_data)
	return np_data


# sklearn自带PCA降维函数 n_com表示保留的特征维数
def sklearn_PCA(np_data, feat=16):
	# 创建PCA对象
	pca = PCA(n_components=feat)
	# 使用PAC对特征数据进行降维
	res_data = pca.fit_transform(np_data)
	# 查看降维后特征的维度
	print('sklearn-PCA 数据降维后维度：{}'.format(res_data.shape))
	# 查看降维后特征数据
	print('sklearn-PCA 数据降维后数据：\n{}'.format(res_data))

	# recon_data = pca.inverse_transform(res_data)
	# print(recon_data.shape)
	# print(recon_data)

	# 查看模型的各个特征向量
	components_ = pca.components_
	print('sklearn-PCA模型的各个特征向量维度：{}'.format(components_.shape))
	print('sklearn-PCA模型的各个特征向量：\n{}'.format(components_))

	# 重构数据 
	reconMat = np.dot(res_data, components_) + pca.mean_
	print('sklearn-PCA 数据重构后维度：{}'.format(reconMat.shape))
	print('sklearn-PCA 数据重构后数据：\n{}'.format(reconMat))

	return res_data, reconMat

# 编写PCA算法
def my_PCA(np_data, feat=16):
	# 计算平均值 数据集为矩阵np_data，每一列代表同一个特征，这里平均值是对每一个特征而言的
	# axis=0 表示按列求均值
	meanVals = np.mean(np_data, axis=0)
	# 减去原始数据的平均值
	meanRemoved = np_data - meanVals
	# 计算协方差矩阵 numpy中的cov函数用于求协方差矩阵
	# rowvar=0，说明传入的数据一行代表一个样本，若非0，说明传入的数据一列代表一个样本
	covMat = np.cov(meanRemoved, rowvar=0)
	# 求特征值、特征矩阵
	# 调用numpy中的线性代数模块linalg中的eig函数，可以直接由协方差矩阵求得特征值和特征向量
	eigVals, eigVects = np.linalg.eig(np.mat(covMat))
	# print(eigVals, eigVects)
	# 对特征值从小到大排序 
	eigValIndice = np.argsort(eigVals)
	# 最大的n个特征值的下标
	n_eigValIndice = eigValIndice[-1:-(feat+1):-1]
	# 最大的n个特征值对应的特征向量  
	n_eigVect = eigVects[:,n_eigValIndice]
	# 低维特征空间的数据  矩阵乘积
	# print(type(meanRemoved), type(n_eigVect))
	lowDDataMat = meanRemoved * n_eigVect
	print('my-PCA 数据降维后维度：{}'.format(lowDDataMat.shape))
	print('my-PCA 数据降维后数据：\n{}'.format(lowDDataMat))
	components_ = n_eigVect.T
	print('mypca模型的各个特征向量维度：{}'.format(components_.shape))
	print('模型的各个特征向量：\n{}'.format(components_))
	# 重构数据 
	reconMat = (lowDDataMat * n_eigVect.T) + meanVals
	print('my-PCA 数据重构后维度：{}'.format(reconMat.shape))
	print('my-PCA 数据重构后数据：\n{}'.format(reconMat))

	return lowDDataMat, reconMat 


# 特征数据标准化
def standardData(data):
	# 创建StandardScaler对象
	sc = StandardScaler()
	sc_data = sc.fit_transform(data)
	return sc_data


def showImage(np_data):
	image = Image.fromarray(np_data)
	image.show()


def plot_gallery(images, titles, h, w, n_row=2, n_col=5):
	plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
	plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
	for i in range(n_row * n_col):
		data = images[i:i+1]
		d = data.astype('float64')
		plt.subplot(n_row, n_col, i + 1)
		plt.imshow(d.reshape((h, w)), cmap=plt.cm.gray)
		# plt.title(titles[i], size=12)
		plt.xticks(())
		plt.yticks(())
	# plt.show()
	plt.savefig('t.png')




def readImage(path):
	image = Image.open(path)
	# image.show()
	np_data = np.array(image)
	# 打印数据发现为三维数据矩阵
	print(np_data.shape)
	# 需要转换成二维数据矩阵及黑白图片
	image = image.convert("L")
	# image.show()
	np_data = np.array(image)
	print(np_data.shape)
	row, col = np_data.shape

	np_data = np.reshape(np_data,(1,row*col))
	# np_data = np.reshape(np_data,(row, col))
	# showImage(np_data)

	return (row, col), np_data


# 主函数
def main():
	# pic_shape, np_data = readImage(r'face2.jpg')
	# showImage(np_data)
	np_data = readData()
	lowDData, components_ = sklearn_PCA(np_data)
	# lowDData, components_ = my_PCA(np_data)
	# print(lowDData.shape)
	# print(lowDData.shape)
	face_titles = ["face %d" % i for i in range(np_data.shape[0])]
	plot_gallery(components_, face_titles, 64, 64)
	# recon = np.reshape(recon, (pic_shape[0], pic_shape[1]))
	# print(recon.shape)
	# showImage(lowDData)


if __name__ == '__main__':
	main()




