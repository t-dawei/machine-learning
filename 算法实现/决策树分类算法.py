#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: T

from sklearn import datasets
from sklearn import model_selection
from sklearn import tree 
import pydotplus
import os
os.environ['PATH'] += os.pathsep + r'D:\graphviz-2\release\bin'

# 读取数据集
def readData():

	iris = datasets.load_iris()

	return iris.data, iris.target


def sklearn_tree(dataSet, target):
	# 数据划分
	train_x, test_x, train_y, test_y = model_selection.train_test_split(dataSet, target, test_size=0.1, random_state=0)
	# 定义决策树分类器 基尼指数
	clf = tree.DecisionTreeClassifier()
	# 训练分类器
	clf.fit(train_x, train_y)

	# 预测测试样本
	pre_y = clf.predict(test_x)

	print('预测结果：\n{}'.format(pre_y))
	print('实际结果：\n{}'.format(test_y))

	saveClf(clf, 'iris')

def saveClf(clf, feature_names):
	dot_data = tree.export_graphviz(clf, out_file=None, feature_names='iris',
										filled=True, rounded=True, special_characters=True)
	graph = pydotplus.graph_from_dot_data(dot_data)

	graph.write_pdf('iris.pdf')
def main():
	dataSet, target = readData()
	sklearn_tree(dataSet, target)

if __name__ == '__main__':
	main()