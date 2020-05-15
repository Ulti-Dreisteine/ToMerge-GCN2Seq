# -*- coding: utf-8 -*-
"""
Created on 2020/3/24 11:29

@Project -> File: pollution-online-data-prediction -> identify_singulars.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import numpy as np


def replace_singulars(data, cols2check, singular_value):
	"""
	替换异常值为np.nan
	
	Example:
	------------------------------------------------------------
	cols2process = code_types
	data = replace_singulars(data, cols2process, singular_value = 0.0)
	data = data[['deviceID', 'time'] + code_types]
	------------------------------------------------------------
	"""
	for col in cols2check:
		data['is_singular_'] = data[col].apply(lambda x: 1 if x == singular_value else 0)
		sing_locs_ = list(data[data['is_singular_'] == 1].index)
		data.loc[sing_locs_, (col,)] = np.nan
	return data



