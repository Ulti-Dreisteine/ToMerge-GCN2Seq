# -*- coding: utf-8 -*-
"""
Created on 2020/4/16 14:43

@Project -> File: pollution-regional-forecast-offline-gcn-training -> step_0_load_raw_and_serialize.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: load raw city data and perform temporal serialzation
"""

import logging

logging.basicConfig(level = logging.INFO)

from lake.decorator import time_cost
import pandas as pd
import numpy as np
import warnings
import datetime
import yaml
import sys, os

sys.path.append('../..')

from lib import proj_dir
from lib import CITY
from mod.tool.dir_file_op import search_files_in_current_dir
from mod.tool.time_conversion import time2stp
from lib.step_0_load_raw_and_process import SELECTED_COLS, ALL_COLS


class LoadAndProcessData(object):
	"""load raw citye data and process"""
	
	def __init__(self):
		self.city: str = CITY
		
	def _load_raw(self):
		"""load city's raw data record"""
		data_dir = os.path.join(proj_dir, 'data/raw')
		
		# search corresponding city file.
		files = search_files_in_current_dir(data_dir, ['{}_cityHour.csv'.format(CITY)])
		if len(files) == 0:
			raise RuntimeError('City data not found ({}).'.format(CITY))
		else:
			if len(files) > 1:
				warnings.warn('Multiple city data files found: {}'.format(CITY))
			file = files[0]
			data = pd.read_csv(os.path.join(data_dir, file))
			self.data = data
	
	def _get_time_stamp(self):
		"""get time stamp"""
		self.data['time'] = self.data['ptime'].apply(lambda x: time2stp(str(int(x)), "%Y%m%d%H"))
		
	def _convert_field_values(self):
		"""field value convertion"""
		# Load code map from json file (because of the existence of Chinese characters in the weather code).
		if 'weather' in SELECTED_COLS:
			with open(os.path.join(proj_dir, 'file/code_map.yml'), 'r', encoding = 'utf-8') as f:
				code_map = yaml.load(f, Loader = yaml.Loader)
			weather_map_dict = code_map['weather_map_dict']
			self.data['weather'] = self.data.loc[:, 'weather'].apply(
				lambda x: weather_map_dict[x] if x in weather_map_dict.keys() else weather_map_dict['晴']
			)
	
	def _build_new_fields(self):
		"""Build new fields"""
		if 'clock_num' in SELECTED_COLS:
			self.data['clock_num'] = self.data['time'].apply(lambda x: x % (24 * 3600) / 3600)
	
		if 'weekday' in SELECTED_COLS:
			if 'date' not in self.data.columns:
				self.data['date'] = self.data['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
			self.data['weekday'] = self.data['date'].apply(lambda x: x.weekday())
	
		if 'month' in SELECTED_COLS:
			if 'date' not in self.data.columns:
				self.data['date'] = self.data['time'].apply(lambda x: datetime.datetime.fromtimestamp(x))
			self.data['month'] = self.data['date'].apply(lambda x: x.month)
	
		self.data = self.data[['time'] + list(ALL_COLS)]
		
	def _replace_singulars(self):
		"""Replce singular values in the fields"""
		self.data.replace(np.nan, 0.0, inplace = True)
		self.data.replace(np.inf, 1.0, inplace = True)
		self.data.reset_index(drop = True, inplace = True)
	
	@time_cost
	def load_and_process_data(self, save = True):
		self._load_raw()
		self._get_time_stamp()
		self._convert_field_values()
		self._build_new_fields()
		self._replace_singulars()
		
		if save:
			self.data.to_csv(os.path.join(proj_dir, 'data/runtime/step_0_data_processed.csv'), index = False)
		
		return self.data


if __name__ == '__main__':
	# %% Load city file and process.
	self = LoadAndProcessData()
	data = self.load_and_process_data(save = True)
	



