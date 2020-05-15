# -*- coding: utf-8 -*-
"""
Created on 2020/4/16 16:08

@Project -> File: pollution-regional-forecast-offline-gcn-training -> step_1_serialize_and_denoise.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import pandas as pd
import sys, os

sys.path.append('../..')

from lib import proj_dir
from lib import HOUR, COLS_BOUNDS
from mod.data_process.normalize_and_denoise import normalize_cols
from mod.data_process.temporal_serialize import DataTemporalSerialization
from lib.step_0_load_raw_and_process import CATEGORICAL_COLS

if __name__ == '__main__':
	# %% Load data.
	data = pd.read_csv(os.path.join(proj_dir, 'data/runtime/step_0_data_processed.csv'))
	
	# %% Data normalization.
	data = normalize_cols(data, COLS_BOUNDS)

	# %% Serialize time stamps in the data.
	start_stp = data['time'].min()
	end_stp = start_stp + (data['time'].max() - data['time'].min()) // HOUR * HOUR

	dts = DataTemporalSerialization(data, start_stp, end_stp, HOUR)
	data, miss_n = dts.temporal_serialize(categorical_cols = CATEGORICAL_COLS)

	# %% Save data.
	data.to_csv(os.path.join(proj_dir, 'data/runtime/step_0_data_srlzd.csv'), index = False)
	


