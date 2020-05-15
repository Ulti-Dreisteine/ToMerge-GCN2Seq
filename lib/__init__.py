# -*- coding: utf-8 -*-
"""
Created on 2020/5/14 16:20

@Project -> File: gcn2seq -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import copy
import sys, os

sys.path.append('../')

from mod.config.config_loader import config

proj_dir, proj_cmap = config.proj_dir, config.proj_cmap

model_params = config.conf
env_params = config.env_conf
test_params = config.test_params

# Env vars here.

# Model params here.
CITY = model_params['CITY']
HOUR = model_params['HOUR']
TARGET_COLS = model_params['TARGET_COLS']
SELECTED_COLS = model_params['SELECTED_COLS']
CATEGORICAL_COLS = model_params['CATEGORICAL_COLS']
COLS_BOUNDS = model_params['COLS_BOUNDS']

NUMERICAL_COLS = [p for p in SELECTED_COLS if p not in CATEGORICAL_COLS]
selected_non_target_cols_ = copy.deepcopy(SELECTED_COLS)
for col in SELECTED_COLS:
	if col in TARGET_COLS:
		selected_non_target_cols_.remove(col)
ALL_COLS = TARGET_COLS + selected_non_target_cols_

# Test params here.

# General functions here.




