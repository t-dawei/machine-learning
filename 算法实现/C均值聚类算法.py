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
# 导入NMI
from NMI import myNMI, sklearnNMI
# 导入xb
from xie_beni import xieBeni
# import math 

# 读取sklearn iris数据集 测试用
def readData():
    iris = datasets.load_iris()
    pd_data = pd.DataFrame(iris.data[:,1:3])
    np_data = np.array(pd_data)
    print('原始数据集的维度：{}'.format(np_data.shape))
    print('原始数据集：\n{}'.format(np_data))
    # 绘图
    graph(np_data)
    return np_data


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
    print(pd_data)
    # 转化为np对象
    np_data = np.array(pd_data)
    labels = np_data[:,2]

    new_lables = []
    for i in labels:
        new_lables.append(int(i))

    # 原始数据图
    graph(data=np_data, color=new_lables, marker='o', title='原始数据图', xlabel='X', ylabel='Y')

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


# 调用sklearn 中的kmean算法 用于与自己写的kmean算法结果对比
def sklearn_kmean(np_data):
    # 将其类别分为3类
    k = KMeans(n_clusters=3)

    # 训练模型
    k.fit(np_data)

    # 质心
    kc = k.cluster_centers_
    print('质心位置：\n{}'.format(kc))

    # 每个样本进行分类
    res_kmeans = k.predict(np_data)
    print('样本分类结果：\n{}'.format(res_kmeans))

    # 画图展示分类结果
    graph(np_data, color=res_kmeans, marker='o', title='sklearn_kmean')


# 随机初始生成聚类中心
def randCluster(np_data, k):
    # 数据集列数
    cols = np_data.shape[1]
    # 构建矩阵初始化
    clucenters = np.mat(np.zeros((k, cols)))
    for col in range(cols):
        # 取这列最值
        mincol = min(np_data[:, col])
        maxcol = max(np_data[:, col])
        # 按列赋值 rand(k, 1) 生成k * 1 矩阵 值为0-1
        clucenters[:, col] = np.mat(mincol + float(maxcol - mincol) * np.random.rand(k, 1))   
    return clucenters


# 计算欧式距离
def distance(vecX, vecY):
    return np.linalg.norm(vecX - vecY)


def myKMeans(np_data, k):
    # 样本数据集行数
    rows = np_data.shape[0]
    # 构造rows * 2 矩阵 col1=数据集所属类别 col2=数据集到聚类中心的距离
    cluster = np.mat(np.zeros((rows, 2)))
    # 随机生成一个数据集的聚类中心 3 * 2 矩阵
    clucenters = randCluster(np_data, k)

    # 设置迭代标志 当flag=false结束
    flag = True
    while flag:
        flag = False
        # 遍历样本集 计算每个样本与聚类中心的欧式距离
        for i in range(rows):
            # 遍历k个聚类中心,获得最短距离
            dist_list = [distance(clucenters[j, :], np_data[i, :]) for j in range(k)]
            # 取最小值的索引
            min_dist = min(dist_list)
            min_index = dist_list.index(min_dist)

            if cluster[i, 0] != min_index:
                # 数据需要更新 还需再次遍历样本集
                flag = True
            # 更新第i个样本所属的类别
            cluster[i, :] = min_index, min_dist
        # 更新聚类中心
        for c in range(k):
            # 取出类别为c的样本数据集
            index = np.nonzero(cluster[:, 0].A == c)[0]
            newClust = np_data[index]
            # 计算newClust各列均值 axis=0表示按列计算
            clucenters[c, :] = np.mean(newClust, axis=0)

    # 作图展示分类结果
    res_predict = [int(x[0]) for x in cluster[:,0].tolist()]
    print(res_predict)
    stand_predict = standardization(res_predict)
    print(stand_predict)
    graph(np_data, color=stand_predict, marker='o', title='my_kmean')
    

    # 构造u -- xb指标使用
    u = np.zeros((rows, k), dtype=np.int)
    for index, item in enumerate(stand_predict):
        u[index][item-1] = 1

    return clucenters, stand_predict, u


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

# 主函数
def main():
    # np_data = readData()
    np_data, labels = readExcel()
    clucenters, stand_predict, u = myKMeans(np_data, 3)
    # sklearn_kmean(np_data)

    # sklearnNMI(labels, stand_predict)
    myNMI(labels, stand_predict)
    
    xieBeni(u, clucenters, np_data)
    


if __name__ == '__main__':
    main()