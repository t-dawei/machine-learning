#!/usr/bin/python
# -*- coding: utf-8 -*-


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
import matplotlib.pyplot as plt

def logisticRegression(x, y):
	'''
	标准化数据
	'''
	# 创建标准化对象
	ss = StandardScaler()
	# 对特征数据标准化
	x_regular = ss.fit_transform(x)

	# 划分数据集和测试集
	x_train, x_test, y_train, y_test = train_test_split(x_regular, y, test_size=0.1)

	'''
	训练模型
	'''
	# 创建逻辑回归对象
	lr = LogisticRegression()
	# 训练
	lr.fit(x_train, y_train)

	'''
	查看模型的效果
	'''
	# 准确率
	r = lr.score(x_train, y_train)
	print('准确率：{}'.format(r))

	# 预测
	y_res = lr.predict(x_test)
	print('正确结果：\n{}'.format(y_test))
	print('预测结果：\n{}'.format(y_res))

	test_validate(x_test, y_test, y_res, lr)

def getData():
	# 导入手写识别数据
	digits = datasets.load_digits()
	x = digits.data
	y = digits.target
	print('data:\n{}'.format(x))
	print('target:\n{}'.format(y))
	return x, y

# 画图对预测值和实际值进行比较
def test_validate(x_test, y_test, y_predict, classifier):
    x = range(len(y_test))
    plt.plot(x, y_test, "ro", markersize=5, zorder=3, label=u"true_v")
    plt.plot(x, y_predict, "go", markersize=8, zorder=2, label=u"predict_v,$R^2$=%.3f" % classifier.score(x_test, y_test))
    # 标签的位置
    plt.legend(loc="upper left")
    plt.xlabel("number")
    plt.ylabel("true?")
    plt.show()

import numpy as np
def gradAscent(x, y):
	# size: n * m
	x_mat = np.mat(x)
	# size: m * 1
	y_mat = np.mat(y)  
	
def myLosisticRegression(x, y):
	pass
def main():
	x, y = getData()
	logisticRegression(x, y)


if __name__ == '__main__':
	main()

	