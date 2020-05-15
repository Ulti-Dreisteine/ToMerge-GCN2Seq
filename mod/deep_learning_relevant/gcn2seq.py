# -*- coding: utf-8 -*-
"""
Created on 2020/4/27 16:30

@Project -> File: pollution-regional-forecast-offline-gcn-training -> gcn2seq.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 基于LSTM的seq2seq模型
"""

import logging

logging.basicConfig(level = logging.INFO)

from torch.nn.parameter import Parameter
import torch.nn.functional as f
from torch.nn import init
from torch import nn
import torch
import sys, os

sys.path.append('../..')

from mod.deep_learning_relevant.gcn import GCN


class GCNLayer(nn.Module):
	"""
	GCN层
	"""
	
	def __init__(self, gcn_block: GCN, seq_len: int):
		super(GCNLayer, self).__init__()
		self.gcn_block = gcn_block
		self.seq_len = seq_len
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""
		:param x: torch.Tensor, shape = (batch_size, nodes_n, seq_len, features_n)
		"""
		x = x.transpose(0, 2)  # shape = (seq_len, nodes_n, batch_size, features_n)
		y = None
		for i in range(self.seq_len):
			x_ = x[i, :, :, :]  # shape = (nodes_n, batch_size, features_n)
			y_ = self.gcn_block(x_)  # shape = (nodes_n, batch_size, hidden_size)
			y_ = y_.unsqueeze(dim = 0)  # shape = (1, nodes_n, batch_size, hidden_size)
			if y is None:
				y = y_
			else:
				y = torch.cat((y, y_), dim = 0)
		
		return y  # shape = (seq_len, nodes_n, batch_size, hidden_size)


class GCN2Seq(nn.Module):
	"""
	GCN2Seq时间序列预测模型
	
	Attributes:
		encoder_seq_len: int, encoder的seq_len参数
		decoder_seq_len: int, decoder的seq_len参数
		with_gcn: bool, 使用使用GCN卷积层
		
		gcn_params: dict, gcn参数
		lstm_params: dict, encoder和decoder参数
		
	"""
	
	def __init__(self, encoder_seq_len: int, decoder_seq_len: int, with_gcn: bool = True):
		super(GCN2Seq, self).__init__()
		self.encoder_seq_len = encoder_seq_len
		self.decoder_seq_len = decoder_seq_len
		self.with_gcn = with_gcn
	
	def init_gcn_layer(self, endo_nodes_n: int, exog_nodes_n: int, features_n: int, L: torch.Tensor, hidden_size: int, layers_n: int):
		"""
		初始化一个GCN单元
		:param endo_nodes_n: int, 内生变量节点数
		:param exog_nodes_n: int, 外生变量节点数
		:param features_n: int, 单个节点上的特征数
		:param L: torch.Tensor, Laplacian矩阵
		:param hidden_size: int, GCN的隐含层输出维数
		:param layers_n: int, GCN隐含层数
		"""
		self.gcn_params = {
			'endo_nodes_n': endo_nodes_n,
			'exog_nodes_n': exog_nodes_n,
			'nodes_n': endo_nodes_n + exog_nodes_n,
			'features_n': features_n,
			'L': Parameter(L, requires_grad = False),  # 静态图Laplacian只是固定参数
			'hidden_size': hidden_size,  # 单个GCN隐含层输出
			'layers_n': layers_n,
			'endo_exog_split_loc': endo_nodes_n * hidden_size  # GCN层输出中endo和exog变量对应输出分界位置
		}
		
		gcn_block_ = GCN(
			nodes_n = self.gcn_params['nodes_n'],
			features_n = self.gcn_params['features_n'],
			L = self.gcn_params['L'],
			hidden_size = self.gcn_params['hidden_size'],
			layers_n = self.gcn_params['layers_n']
		)
		
		self.gcn_layer = GCNLayer(gcn_block_, self.encoder_seq_len)  # 由同一个GCN block生成的GCN层
	
	def init_encoders_n_decoders(self, hidden_size: int, layers_n: int, bias: bool, batch_first: bool = True, dropout: float = 0.0):
		"""
		生成内生变量和外生变量encoder层
		"""
		self.lstm_params = {
			'hidden_size': hidden_size,
			'layers_n': layers_n,
			'bias': bias,
			'batch_first': batch_first,
			'dropout': dropout
		}
		
		self.exog_encoder = nn.LSTM(
			input_size = self.gcn_params['exog_nodes_n'] * self.gcn_params['hidden_size'],  # gcn外生变量输出部分
			hidden_size = self.lstm_params['hidden_size'],
			num_layers = self.lstm_params['layers_n'],
			bias = self.lstm_params['bias'],
			batch_first = self.lstm_params['batch_first'],
			dropout = self.lstm_params['dropout']
		)
		
		self.endo_encoder = nn.LSTM(
			input_size = self.gcn_params['endo_nodes_n'] * self.gcn_params['hidden_size'] + self.exog_encoder.hidden_size,
			# gcn内生变量输出部分+外生变量encoder输出部分
			hidden_size = self.lstm_params['hidden_size'],
			num_layers = self.lstm_params['layers_n'],
			bias = self.lstm_params['bias'],
			batch_first = self.lstm_params['batch_first'],
			dropout = self.lstm_params['dropout']
		)
		
		self.exog_decoder = nn.LSTM(
			input_size = self.exog_encoder.hidden_size,
			hidden_size = self.lstm_params['hidden_size'],
			num_layers = self.lstm_params['layers_n'],
			bias = self.lstm_params['bias'],
			batch_first = self.lstm_params['batch_first'],
			dropout = self.lstm_params['dropout']
		)
		
		self.endo_decoder = nn.LSTM(
			input_size = self.endo_encoder.hidden_size + self.exog_decoder.hidden_size,
			hidden_size = self.lstm_params['hidden_size'],
			num_layers = self.lstm_params['layers_n'],
			bias = self.lstm_params['bias'],
			batch_first = self.lstm_params['batch_first'],
			dropout = self.lstm_params['dropout']
		)
		
		self._init_decoder_out_fcs()
	
	def _init_decoder_out_fcs(self):
		# 外生变量.
		self.exog_decode_out_fc = nn.Linear(
			self.decoder_seq_len * self.lstm_params['hidden_size'],
			self.gcn_params['exog_nodes_n'] * self.decoder_seq_len * self.gcn_params['features_n'],
			bias = False)
		init.normal_(self.exog_decode_out_fc.weight)
		
		# 内生变量.
		self.endo_decode_out_fc = nn.Linear(
			2 * self.decoder_seq_len * self.lstm_params['hidden_size'],
			self.gcn_params['endo_nodes_n'] * self.decoder_seq_len * self.gcn_params['features_n'],
			bias = False)
		init.normal_(self.endo_decode_out_fc.weight)
	
	def forward(self, x: torch.Tensor):
		# x.shape = (batch_size, nodes_n, seq_len, features_n)
		batch_size = x.shape[0]
		
		# 计算GCN layer层输出.
		if self.with_gcn:
			gcn_layer_out_ = self.gcn_layer(x)  # shape = (seq_len, nodes_n, batch_size, hidden_size)
		else:
			gcn_layer_out_ = x.transpose(0, 2)  # shape = (seq_len, nodes_n, batch_size, features_n)
		
		# 计算encoder层输出, 并分为内生变量和外生变量部分.
		gcn_layer_out_ = gcn_layer_out_.transpose(0, 2)  # shape = (batch_size, nodes_n, seq_len, hidden_size)
		gcn_layer_out_ = gcn_layer_out_.transpose(1, 2)  # shape = (batch_size, seq_len, nodes_n, hidden_size)
		gcn_layer_out_ = gcn_layer_out_.reshape(batch_size, self.encoder_seq_len, self.gcn_params['nodes_n'] * self.gcn_params['hidden_size'])
		
		endo_encoder_in_ = gcn_layer_out_[:, :, :self.gcn_params['endo_exog_split_loc']]  # shape = (batch_size, seq_len, endo_nodes_n * hidden_size)
		exog_encoder_in_ = gcn_layer_out_[:, :, self.gcn_params['endo_exog_split_loc']:]  # shape = (batch_size, seq_len, exog_nodes_n * hidden_size)
		
		# 计算encoder编码.
		# 计算exog、endo的encoder输出.
		
		# exog_encoder_out_.shape = (batch_size, seq_len, hidden_size)
		# h_exog_encoder_.shape = c_exog_encoder_.shape = (layers_n, batch_size, hidden_size)
		exog_encoder_out_, (h_exog_encoder_, c_exog_encoder_) = self.exog_encoder(exog_encoder_in_)
		
		# endo_encoder_out_.shape = (batch_size, seq_len, hidden_size)
		# h_endo_encoder_.shape = c_endo_encoder_.shape = (layers_n, batch_size, hidden_size)
		endo_encoder_in_ = torch.cat((endo_encoder_in_, exog_encoder_out_), dim = 2)
		endo_encoder_out_, (h_endo_encoder_, c_endo_encoder_) = self.endo_encoder(endo_encoder_in_)
		
		# 计算decoder解码.
		exog_decoder_out_ = None
		y_ = exog_encoder_out_[:, -1:, :]
		h_, c_ = h_exog_encoder_, c_exog_encoder_
		for i in range(self.decoder_seq_len):
			y_, (h_, c_) = self.exog_decoder(y_, (h_, c_))
			if i == 0:
				exog_decoder_out_ = y_
			else:
				exog_decoder_out_ = torch.cat((exog_decoder_out_, y_), dim = 1)
		
		endo_decoder_out_ = None
		y_ = endo_encoder_out_[:, -1:, :]
		h_, c_ = h_endo_encoder_, c_endo_encoder_
		for i in range(self.decoder_seq_len):
			y_, (h_, c_) = self.endo_decoder(torch.cat((y_, exog_decoder_out_[:, i: i + 1, :]), dim = 2), (h_, c_))
			if i == 0:
				endo_decoder_out_ = y_
			else:
				endo_decoder_out_ = torch.cat((endo_decoder_out_, y_), dim = 1)
		
		# %% 输出结果经过FC变换.
		y_endo_decoded = endo_decoder_out_.reshape(batch_size, self.decoder_seq_len * self.lstm_params['hidden_size'])
		y_exog_decoded = exog_decoder_out_.reshape(batch_size, self.decoder_seq_len * self.lstm_params['hidden_size'])
		
		y_exog_out_ = self.exog_decode_out_fc(y_exog_decoded)  # shape = (batch_size, nodes_n, seq_len, features_n)
		y_endo_out_ = self.endo_decode_out_fc(
			torch.cat((y_endo_decoded, y_exog_decoded), dim = 1))  # shape = (batch_size, nodes_n, seq_len, features_n)
		
		# %% 激活函数.
		y = f.softplus(torch.cat((y_endo_out_, y_exog_out_), dim = 1))
		y = y.reshape(batch_size, self.gcn_params['nodes_n'], self.decoder_seq_len)  # shape = (batch_size, nodes_n, pred_seq_len)
		y = y.transpose(1, 2)  # shape = (batch_size, pred_seq_len, nodes_n)
		
		return y


if __name__ == '__main__':
	# %% Test params.
	from lib.step_1_train_model import NODES_N, FEATURES_N, A, ENDO_NODES_N, EXOG_NODES_N
	from lib.step_1_train_model import hist_len, pred_len
	from lib.step_1_train_model import trainloader, verifloader, L
	
	x_train, y_train = [p for p in trainloader][0]
	
	# x.shape = (batch_size, nodes_n, seq_len, features_n)
	# y.shape = (batch_size, pred_len * nodes_n * features_n)
	x, y = x_train, y_train
	encoder_seq_len, decoder_seq_len = hist_len + 1, pred_len
	
	# %% Test GCN2Seq.
	self = GCN2Seq(encoder_seq_len, decoder_seq_len)
	
	gcn_params = {
		'endo_nodes_n': ENDO_NODES_N,
		'exog_nodes_n': EXOG_NODES_N,
		'features_n': FEATURES_N,
		'L': L,
		'hidden_size': 4,
		'layers_n': 2
	}
	encoders_params = {
		'hidden_size': 4,
		'layers_n': 2,
		'bias': False,
		'batch_first': True,
		'dropout': 0.0
	}
	
	self.init_gcn_layer(**gcn_params)
	self.init_encoders_n_decoders(**encoders_params)
	
	# %% Forward.
	y_pred = self(x)
