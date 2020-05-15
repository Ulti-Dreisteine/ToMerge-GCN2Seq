# -*- coding: utf-8 -*-
"""
Created on 2020/1/20 上午10:19

@Project -> File: pollution-pdmc-relevance-analyzer -> normalize.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据标准化
"""

import numpy as np


def min_max_normalize(x):
	"""
	对np.array每列按照min-max归一化
	:param x: np.array, 待归一化的数据（表）
	:return: np.array, 待归一化的数据（表）
	:return: (v_min, v_max): 当x只有一列时返回列归一化前的最小最大值
	"""
	
	try:
		if x.shape[1] == 1:
			v_min, v_max = np.min(x), np.max(x)
			x = (x - v_min) / (v_max - v_min)
			return x, (v_min, v_max)
		elif x.shape[1] > 1:
			for i in range(x.shape[1]):
				v_min, v_max = np.min(x[:, i]), np.max(x[:, i])
				x[:, i] = (x[:, i] - v_min) / (v_max - v_min)
			return x
		else:
			raise RuntimeError('归一化失败')
	except Exception as e:
		raise RuntimeError(e)


def min_max_normalize2(x):
	"""
	对np.array每列按照min-max归一化
	:param x: np.array, 待归一化的数据（表）
	:return: np.array, 待归一化的数据（表）
	:return: (v_min, v_max)
	"""
	
	eps = 1e-10
	try:
		if x.shape[1] == 1:
			v_min, v_max = np.min(x), np.max(x)
			x = (x - v_min) / (v_max - v_min + eps)
			return x, (v_min, v_max)
		elif x.shape[1] > 1:
			v_min, v_max = np.min(x, axis = 0), np.max(x, axis = 0)
			x = (x - v_min) / (v_max - v_min + eps)
			return x, (v_min, v_max)
		else:
			raise RuntimeError('归一化失败')
	except Exception as e:
		raise RuntimeError(e)


def standard_normalize(x, eps = 1e-6):
	"""
	对np.array标准化处理
	:param x: np.array, 待归一化的数据
	:param eps: 防止除数为0
	:return: np.array, 待归一化的数据
	:return: (mean, std): 返回均值和标准差
	"""
	
	try:
		if x.shape[1] == 1:
			mean = np.mean(x)
			std = np.std(x)
			x = (x - mean) / (std + eps)
			return x, (mean, std)
		elif x.shape[1] > 1:
			mean = np.mean(x, axis = 0)
			std = np.mean(x, axis = 0)
			x = (x - mean) / (std + eps)
			return x, (mean, std)
	except Exception as e:
		raise RuntimeError(e)



