#!/usr/bin/env python
#-*- coding:utf-8 -*-
# @auther: T

import matplotlib.pyplot as plt
import numpy as np
'''
plt.scatter(x, y, s, c, marker, cmap, linewidth)
x: 横坐标数组、
y: 轴坐标数组
s: size
c: color [c, b, y, g, k]
marker: 形状 ['.', 'o', '^', 'V']
cmap: colormap 对象
linewidth: 形状边缘线宽
'''



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

	
# 画两个子图
def drawScatter2(x, y):
	# 获取画布对象
	fig = plt.figure()

	# 通过画布 将画图 分割成4个字图 2*2
	ax1 = fig.add_subplot(121)
	ax1.set_title('1')
	plt.xlabel('x1')
	plt.ylabel('y1')


	ax2 = fig.add_subplot(122)
	ax2.set_title('2')
	plt.xlabel('x2')
	plt.ylabel('y2')

	ax1.scatter(x=x, y=y, s=10, c='r', marker='o', linewidth=1)
	ax2.scatter(x=x, y=y, s=10, c='y', marker='v', linewidth=2)

	plt.subplots_adjust(bottom=0.2, left=0.1, right=0.95, top=0.7, hspace=0.3, wspace=0.4)

	plt.show()

# 不画子图
def drawScatter(x, y):

	# plt 本身就是最大的子图
	plt.xlabel('x1')
	plt.ylabel('y1')
	plt.scatter(x=x, y=y, s=10, c='r', marker='o', linewidth=1)
	plt.show()

def randomData():
	np_x = np.random.randint(1, 20, size=20)
	np_y = np.random.randint(1, 20, size=20)
	print(np_x, np_y)
	return np_x, np_y

def main():
	x, y = randomData()
	drawScatter(x, y)


if __name__ == '__main__':
	main()

