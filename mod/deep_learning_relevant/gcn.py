# -*- coding: utf-8 -*-
"""
Created on 2020/4/24 10:28

@Project -> File: gcn -> gcn.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

from torch import nn
from torch.nn import init
import numpy as np
import torch
import sys, os

sys.path.append('../..')


class GCN(nn.Module):
	"""
	GCN网络
	
	Attributes:
		nodes_n: int, 图网络中变量节点数
		features_n: int, 每个节点上的特征数
		L: torch.Tensor, Laplacian矩阵
		hidden_size: int, 特征隐含层维数
		layers_n: int, 所有神经元层数，必须大于等于2
	"""
	
	def __init__(self, nodes_n: int, features_n: int, L: torch.Tensor, hidden_size: int, layers_n: int):
		try:
			assert layers_n > 1
		except:
			raise ValueError('layers_n is not greater than 1')  # 一个神经网络里至少要有两层神经元
			
		super(GCN, self).__init__()
		self.nodes_n = nodes_n
		self.features_n = features_n
		self.L = L  # .to_sparse()  # 转为二维稀疏张量
		self.hidden_size = hidden_size
		self.layers_n = layers_n
		
		self.hidden_layers_n_ = self.layers_n - 1
		
		self._init_linear_modules()
		self._init_bn_modules()
		self._init_bn_in()
		
	def _init_linear_modules(self):
		self.fcs = []
		for i in range(self.hidden_layers_n_):
			if i == 0:
				fc_ = nn.Linear(self.features_n, self.hidden_size, bias = False)
			else:
				fc_ = nn.Linear(self.hidden_size, self.hidden_size, bias = False)
			init.uniform_(fc_.weight)
			setattr(self, 'fc_{}'.format(i), fc_)
			self.fcs.append(fc_)
	
	def _init_bn_modules(self):
		self.bns = []
		for i in range(self.hidden_layers_n_):
			bn_ = nn.BatchNorm1d(self.nodes_n * self.hidden_size)
			setattr(self, 'bn_{}'.format(i), bn_)
			self.bns.append(bn_)
	
	def _init_bn_in(self):
		self.bn_in = nn.BatchNorm1d(self.nodes_n * self.features_n)
	
	def forward(self, x: torch.Tensor):
		"""
		前向计算
		:param x: torch.Tensor, shape = (nodes_n, batch_size, features_n)
		"""
		batch_size_ = x.shape[1]
		x = x.transpose(0, 1)
		x = x.reshape(batch_size_, self.nodes_n * self.features_n)
		x = self.bn_in(x)
		x = x.reshape(batch_size_, self.nodes_n, self.features_n)
		x = x.transpose(0, 1)  # shape = (nodes_n, batch_size, features_n)
		
		h = x
		for i in range(self.layers_n - 1):
			if i == 0:
				size_ = self.features_n
			else:
				size_ = self.hidden_size
				
			# 计算 a = L \cdot h.
			h = h.reshape(self.nodes_n, batch_size_ * size_)
			h = torch.matmul(self.L, h)  # shape = (nodes_n, batch_size * size_)
# 			h = torch.hspmm(self.L, h).to_dense()
			
			# 计算 a \cdot w.
			h.reshape(self.nodes_n, batch_size_, size_)
			h = h.reshape(self.nodes_n * batch_size_, size_)
			h = self.fcs[i](h)  # shape = (nodes_n * batch_size, hidden_size)
			h = h.reshape(self.nodes_n, batch_size_, self.hidden_size)
			
			# bn操作.
			h = h.transpose(0, 1)
			h = h.reshape(batch_size_, self.nodes_n * self.hidden_size)
			h = self.bns[i](h)
			h = h.reshape(batch_size_, self.nodes_n, self.hidden_size)
			h = h.transpose(0, 1)
			
			# 激活函数.
# 			h = torch.sigmoid(h)  # shape = (nodes_n, batch_size, hidden_size)
			h = torch.relu(h)
			
		return h
	

if __name__ == '__main__':
	# %% 测试参数.
	import pandas as pd
	from lib import proj_dir

	data = pd.read_csv(os.path.join(proj_dir, 'data/runtime/step_0_data_srlzd.csv'))
	cols = ['pm25', 'pm10', 'sd']
	data = data.iloc[: 1000][cols]
	x = torch.from_numpy(np.array(data, dtype = np.float32))
	x = x.unsqueeze(dim = 2)    # 扩充为(samples_n, nodes_n, features_n)的形式
	x = x.transpose(0, 1)       # 变换维度为(nodes_n, samples_n, features_n)

	nodes_n, samples_n, features_n = x.shape
	L = torch.from_numpy(
		np.array([
			[2, -1, -1],
			[-1, 2, -1],
			[-1, -1, 2]
		], dtype = np.float32)
	)

	# %% 测试GCN.
	hidden_size = 8
	layers_n = 3
	self = GCN(nodes_n, features_n, L, hidden_size, layers_n)
	y = self(x)
	
	# # %% 测试torch.hspmm()
	# L = torch.tensor(
	# 	[
	# 		[2, 0, 0],
	# 		[0, 2, 0],
	# 		[1, 0, 2]
	# 	]
	# ).to_sparse()
	# h = torch.tensor(
	# 	[
	# 		[1],
	# 		[1],
	# 		[2]
	# 	]
	# )
	#
	# a = torch.hspmm(L, h).to_dense()



