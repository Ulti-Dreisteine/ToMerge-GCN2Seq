# -*- coding: utf-8 -*-
"""
Created on 2020/5/15 17:07

@Project -> File: gcn2seq -> __init__.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: initialize some modeling data & params
"""

import logging

logging.basicConfig(level = logging.INFO)

from scipy.ndimage.interpolation import shift
from torch.utils.data import DataLoader
import torch.utils.data as Data
import pandas as pd
import numpy as np
import torch
import sys, os

sys.path.append('../..')

from lib import proj_dir

data = pd.read_csv(os.path.join(proj_dir, 'data/runtime/step_0_data_srlzd.csv'))

ENDO_COLS = ['pm25', 'pm10', 'so2', 'co', 'no2', 'o3']  # endogeneous variables
EXOG_COLS = ['ws', 'temp', 'sd']                        # exogeneous variables
COLS = ENDO_COLS + EXOG_COLS

NODES_N = len(COLS)     # variables number
ENDO_NODES_N = len(ENDO_COLS)
EXOG_NODES_N = len(EXOG_COLS)
FEATURES_N = 1          # feature number must be 1, because the value of each node at one time is scalar
data = data[COLS]       # the modeling data is a subset of the original

# Adjacency matrix.
# Got by nonlinear time series analysis & actual experience.
A = np.array([
	[0, 1, 1, 1, 1, 1, 0, 0, 0],
	[1, 0, 1, 1, 1, 1, 0, 0, 0],
	[1, 1, 0, 1, 1, 1, 0, 0, 0],
	[1, 1, 1, 0, 1, 1, 0, 0, 0],
	[1, 1, 1, 1, 0, 1, 0, 0, 0],
	[1, 1, 1, 1, 1, 0, 0, 0, 0],
	[1, 1, 1, 1, 1, 1, 0, 0, 1],
	[0, 0, 1, 0, 1, 1, 1, 0, 1],
	[1, 1, 1, 1, 1, 1, 0, 0, 0],
], dtype = np.float32)


def build_total_samples_tensors(samples_n: int, hist_len: int, pred_len: int, nodes_n: int, features_n: int) -> (
torch.Tensor, torch.Tensor):
	"""
	Build total samples for X and Y
	:param samples_n: int, samples_len
	:param hist_len: int, history features time len (the current is not included), so the time steps for sample X is hist_len + 1
	:param pred_len: int, predicted time steps (the current is not included)
	:param nodes_n: int, nodes number of a single variable
	:param features_n: int, features number on eacher node
	"""
	total_arr = np.array(data, dtype = np.float32)  # shape = (samples_len, nodes_n * features_n)
	for i in range(hist_len + pred_len):
		d = data.copy()
		for col in COLS:
			d[col] = shift(d[col], -1 * (i + 1))  # 向上平移
		arr = np.array(d, dtype = np.float32)  # shape = (samples_len, nodes_n * features_n)
		total_arr = np.hstack((total_arr, arr))  # shape = (samples_len, total_blocks_n * nodes_n * features_n)
	
	x = total_arr[: samples_n, :-(nodes_n * pred_len)]  # shape = (samples_n, hist_len * nodes_n * features_n)
	y = total_arr[: samples_n, -(nodes_n * pred_len):]  # shape = (samples_n, pred_len * nodes_n * features_n)
	
	blocks_n = hist_len + 1
	x = torch.from_numpy(x)
	x = x.reshape(samples_n, blocks_n, nodes_n, features_n)
	x = x.transpose(0, 1)
	x = x.transpose(1, 2)  # shape = (blocks_n, nodes_n, samples_n, features_n)
	y = torch.from_numpy(y)  # shape = (samples_n, pred_len * nodes_n * features_n)
	
	return x, y


def build_train_verif_loader(x: torch.Tensor, y: torch.Tensor, train_ratio: float, batch_size: int):
	# x.shape = (blocks_n, nodes_n, samples_n, features_n)
	# y.shape = (samples_n, pred_len * nodes_n)
	# shuffle.
	idxes = list(range(x.shape[2]))
	idxes = np.random.permutation(idxes)
	x, y = x[:, :, idxes, :], y[idxes, :]
	
	sep_loc = int(x.shape[2] * train_ratio)
	x_train, y_train = x[:, :, :sep_loc, :], y[:sep_loc, :]
	x_verif, y_verif = x[:, :, sep_loc:, :], y[sep_loc:, :]
	
	train_dataset = Data.TensorDataset(x_train.transpose(0, 2), y_train)  # x.shape = (batch_size, nodes_n, blocks_n, features_n)
	trainloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False)
	verif_dataset = Data.TensorDataset(x_verif.transpose(0, 2), y_verif)  # x.shape = (batch_size, nodes_n, blocks_n, features_n)
	verifloader = DataLoader(verif_dataset, batch_size = batch_size, shuffle = False)
	
	return trainloader, verifloader


def gen_laplacian_matrix(A: np.ndarray) -> torch.Tensor:
	D = np.diag(np.sum(A, axis = 1))
	D_inv = np.linalg.inv(D)
	L = D - A
	L_rw = np.dot(D_inv, L)
	L = torch.from_numpy(L_rw)  # Randow-walk Laplacian is adopted here.
	return L


# ============ Modeling Data & Params ============
# Laplacian matrix for building GCN layer later.
L = gen_laplacian_matrix(A)

# Build total samples X & Y.
hist_len = 5  # **current stp is not included, so the corresponding dim of X should be hist_len + 1
pred_len = 8  # **current stp is not included，pred time steps
samples_n = 20000
X, Y = build_total_samples_tensors(samples_n, hist_len, pred_len, NODES_N, FEATURES_N)

# Build train & verify set data loader.
train_ratio = 0.8
batch_size = 2000
trainloader, verifloader = build_train_verif_loader(X, Y, train_ratio, batch_size)








