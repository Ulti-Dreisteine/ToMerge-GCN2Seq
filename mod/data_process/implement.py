# -*- coding: utf-8 -*-
"""
Created on 2020/2/19 11:26

@Project -> File: pollution-online-data-prediction -> implement.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 数据填补
"""

import pandas as pd
import numpy as np
import sys

sys.path.append('../..')

from mod.data_process import search_nearest_neighbors_in_list


def data_implement(data: pd.DataFrame, fields2process: list = None):
	"""
	数据填补
	:param data: pd.DataFrame, 待填补数据表
	:param fields2process: list of strs, 需要进行缺失填补的字段
	
	Example:
	------------------------------------------------------------
	data = data_implement(data, code_types)
	------------------------------------------------------------
	"""
	data = data.copy()
	
	if fields2process is None:
		fields2process = data.columns
	
	# 逐字段缺失值填补.
	for field in fields2process:
		print('Implementing field "{}"'.format(field))
		values = list(data.loc[:, field])
		
		total_idxs = list(range(len(values)))
		ineffec_idxs = list(np.argwhere(np.isnan(values)).flatten())
		effec_idxs = list(set(total_idxs).difference(set(ineffec_idxs)))
		
		ineffec_idxs.sort()
		effec_idxs.sort()
		
		for idx in ineffec_idxs:
			neighbor_effec_idxs = search_nearest_neighbors_in_list(effec_idxs, idx)
			value2implement = np.mean(data.loc[neighbor_effec_idxs, field])
			data.loc[idx, field] = value2implement
	print('\n')
	
	return data
	
	



