# -*- coding: utf-8 -*-
"""
Created on 2020/1/22 下午2:58

@Project -> File: algorithm-tools -> normalize_and_denoise.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据归一化和去噪
"""

from math import factorial
import pandas as pd
import numpy as np
import warnings


def normalize_cols(data, cols_bounds: dict) -> pd.DataFrame:
	"""
	数据表按照列进行归一化.
	:param data: pd.DataFrame, 待归一化数据表
	:param cols_bounds: dict, 各字段设定归一化的最小最大值边界
	:return: data: pd.DataFrame, 归一化后的数据表
	
	Note:
		1. 只有cols_bounds中选中的列才会进行归一化;
		2. 如果实际数据中该字段值超出界限则需要调整cols_bounds中的上下界设置;
		
	Example:
	------------------------------------------------------------
	import pandas as pd
	import sys
	import os
	
	sys.path.append('../..')
	
	from lib import proj_dir
	
	data = pd.read_csv(os.path.join(proj_dir, 'data/provided/weather/data_srlzd.csv'))
	
	# %% 归一化
	cols_bounds = {
		'pm25': [0, 600],
		'pm10': [0, 1400],
		'so2': [0, 600],
		'co': [0, 8],
		'no2': [0, 250],
		'o3': [0, 400],
		'aqi': [0, 1300],
		'weather': [0, 19],
		'ws': [0, 10],
		'wd': [0, 18],
		'temp': [-40, 60],
		'sd': [0, 110],
		'month': [0, 12],
		'weekday': [0, 7],
		'clock_num': [0, 24]}
	
	data_nmlzd = normalize_cols(data, cols_bounds)
	------------------------------------------------------------
	"""
	data = data.copy()
	for col in cols_bounds.keys():
		if col in data.columns:
			bounds = cols_bounds[col]
			col_min, col_max = data[col].min(), data[col].max()
			
			print('Normalizing column {}, min_value = {}, bounds = {}'.format(col, col_min, bounds))
			if (col_min < bounds[0]) | (col_max > bounds[1]):
				warnings.warn(
					"var bounds error: column {}'s actual bounds are [{}, {}], while the bounds are set to [{}, {}]".format(
						col, col_min, col_max, bounds[0], bounds[1]
					)
				)
				data.loc[data[col] < bounds[0], col] = bounds[0]
				data.loc[data[col] > bounds[1], col] = bounds[1]
			
			data[col] = data[col].apply(lambda x: (x - bounds[0]) / (bounds[1] - bounds[0]))
	
	return data


def savitzky_golay(y, window_size, order, deriv = 0, rate = 1):
	"""
	savitzky_golay滤波
	"""
	
	try:
		window_size = np.abs(np.int(window_size))
		order = np.abs(np.int(order))
	except Exception:
		raise ValueError("window_size and order have to be of type int")
	
	if window_size % 2 != 1 or window_size < 1:
		raise TypeError("window_size size must be a positive odd number")
	if window_size < order + 2:
		raise TypeError("window_size is too small for the polynomials order")
	
	order_range = range(order + 1)
	half_window = (window_size - 1) // 2
	
	# 预计算系数
	b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
	m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
	
	# pad the signal at the extremes with values taken from the signal itself
	firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
	lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
	y = np.concatenate((firstvals, y, lastvals))
	
	return np.convolve(m[::-1], y, mode = 'valid')


def denoise_cols(data, cols2denoise, window_size: int = None, order: int = None, params: dict = None) -> pd.DataFrame:
	"""
	对数据表选中列进行去噪.
	:param data: pd.DataFrame, 待去噪数据表.
	:param cols2denoise: list of strs, 选中用于去噪的字段
	:param window_size: int > 3, Savitzky-Golay滤波双侧窗口长度
	:param order: int > 1, Savitzky-Golay滤波阶数
	:param params: dict, 各字段滤波参数
	
	Example:
	------------------------------------------------------------
	# %% 去噪
	cols2denoise = ['pm25', 'pm10', 'so2', 'no2', 'o3', 'co', 'aqi', 'ws', 'temp', 'sd']
	window_size = 3
	order = 1
	
	data_denoised = denoise_cols(data_nmlzd, cols2denoise, window_size, order)
	------------------------------------------------------------
	"""
	data = data.copy()
	
	if not params:
		if window_size < 2 * order + 1:
			raise ValueError('window_size应不小于2 * order + 1.')
	
	for col in cols2denoise:
		if col in data.columns:
			y = np.array(data[col])
			if params:
				data[col] = savitzky_golay(y, **params[col])
			else:
				data[col] = savitzky_golay(y, window_size = window_size, order = order)
	return data
	
	
	
	
	



