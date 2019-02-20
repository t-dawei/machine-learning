#!/usr/bin/python
# -*- coding: utf-8 -*-
# @author: 
import numpy as np 

'''
对象属性 
'''
# 矩阵维度
np.shape
# 元素个数
np.size
# 元素类型
np.dtype

'''
创建numpy对象
'''
# 创建 n * m 的矩阵 元素为0 
np.zeros([n,m])
# 创建 n * m 的矩阵 元素为1
np.ones([n, m])


'''
计算两个向量的欧式距离
'''
np.linalg.norm(vecX - vecY)

'''
获取矩阵或者list按从小到大的索引 0表示按列求
'''
np.argsort(vec, axis=0)
