# -*- coding: utf-8 -*-
"""
Created on 2020/4/20 14:09

@Project -> File: gcn -> nn.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: NN网络
"""

import logging

logging.basicConfig(level = logging.INFO)

from torch import nn
from torch.nn import init
import torch.nn.functional as f
import torch
import sys, os

sys.path.append('../..')


class NN(nn.Module):
	"""神经网络前向模块"""
	
	def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
		super(NN, self).__init__()
		self.input_size = input_size
		self.hidden_sizes = hidden_sizes
		self.output_size = output_size
		
		self.bn_in = nn.BatchNorm1d(self.input_size)
		
		self.fc_0 = nn.Linear(self.input_size, self.hidden_sizes[0])
		self._init_layer(self.fc_0)
		self.bn_0 = nn.BatchNorm1d(self.hidden_sizes[0])
		
		self.fcs = []
		self.bns = []
		for i in range(len(hidden_sizes) - 1):
			fc_i = nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i + 1])
			setattr(self, 'fc_{}'.format(i + 1), fc_i)
			self._init_layer(fc_i)
			bn_i = nn.BatchNorm1d(self.hidden_sizes[i + 1])
			setattr(self, 'bn_{}'.format(i + 1), bn_i)
			self.fcs.append(fc_i)
			self.bns.append(bn_i)
		
		self.fc_out = nn.Linear(self.hidden_sizes[-1], self.output_size)
		self._init_layer(self.fc_out)
		self.bn_out = nn.BatchNorm1d(self.output_size)
	
	@staticmethod
	def _init_layer(layer):
		init.normal_(layer.weight)  # 使用这种初始化方式能降低过拟合
		init.normal_(layer.bias)
	
	def forward(self, x):
		x = self.bn_in(x)
		x = self.fc_0(x)
		x = self.bn_0(x)
		x = torch.tanh(x)
		
		for i in range(len(self.fcs)):
			x = self.fcs[i](x)
			x = self.bns[i](x)
			x = torch.tanh(x)
		
		x = self.fc_out(x)
		x = self.bn_out(x)
		x = f.softplus(x)
		
		return x


if __name__ == '__main__':
	pass



