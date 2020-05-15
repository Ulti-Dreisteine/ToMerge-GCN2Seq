# -*- coding: utf-8 -*-
"""
Created on 2020/4/30 13:51

@Project -> File: pollution-regional-forecast-offline-gcn-training -> gcn_seq2seq.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: GCN + Seq2Seq时间序列预测模型
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

from mod.neural_network.gcn import GCN, GCNSeqLayer
from mod.neural_network.lstm import Seq2Seq


class GCNSeq2Seq(nn.Module):
	"""
	GCN + Seq2Seq时间序列预测模型
	"""
	
	def __init__(self, endo_nodes_n: int, exog_nodes_n: int, features_n: int, L: torch.Tensor, encoder_seq_len: int, decoder_seq_len: int):
		super(GCNSeq2Seq, self).__init__()
		self.endo_nodes_n = endo_nodes_n
		self.exog_nodes_n = exog_nodes_n
		self.nodes_n = endo_nodes_n + exog_nodes_n
		self.features_n = features_n
		self.encoder_seq_len = encoder_seq_len
		self.decoder_seq_len = decoder_seq_len
		self.L = Parameter(L, requires_grad = False)
	
	def init_gcn_layer(self, hidden_size: int, layers_n: int):
		gcn_ = GCN(self.nodes_n, self.features_n, self.L, hidden_size, layers_n)
		self.gcn_layer = GCNSeqLayer(gcn_, self.encoder_seq_len)  # output_size = (seq_len, nodes_n, batch_size, hidden_size)
		self.gcn_params = {
			'endo_exog_split_loc': self.endo_nodes_n * hidden_size,
			'hidden_size': hidden_size
		}
	
	def init_exog_seq2seq(self, hidden_size: int, layers_n: int, dropout: float = 0.0):
		exog_seq2seq = Seq2Seq()
		exog_seq2seq.init_encoder(
			input_size = self.exog_nodes_n * self.gcn_params['hidden_size'],
			hidden_size = hidden_size,
			layers_n = layers_n,
			dropout = dropout,
		)
		exog_seq2seq.init_decoder(
			input_size = exog_seq2seq.encoder.hidden_size,  # ** 这里没有接入天气预报数据
			decoder_seq_len = self.decoder_seq_len,
			dropout = dropout
		)
		self.exog_seq2seq = exog_seq2seq
		self.exog_seq2seq_params = {
			'hidden_size': hidden_size
		}
	
	def init_endo_seq2seq(self, hidden_size: int, layers_n: int, dropout: float = 0.0):
		endo_seq2seq = Seq2Seq()
		endo_seq2seq.init_encoder(
			input_size = self.endo_nodes_n * self.gcn_params['hidden_size'] + self.exog_seq2seq.encoder.hidden_size,
			hidden_size = hidden_size,
			layers_n = layers_n,
			dropout = dropout,
		)
		endo_seq2seq.init_decoder(
			input_size = endo_seq2seq.encoder.hidden_size + self.exog_seq2seq.decoder.hidden_size,
			decoder_seq_len = self.decoder_seq_len,
			dropout = dropout
		)
		self.endo_seq2seq = endo_seq2seq
		self.endo_seq2seq_params = {
			'hidden_size': hidden_size
		}
		self._init_decoder_out_fc_layers()
		
	def _init_decoder_out_fc_layers(self):
		# 外生变量.
		self.exog_decode_out_fc = nn.Linear(
			self.decoder_seq_len * self.exog_seq2seq_params['hidden_size'],
			self.exog_nodes_n * self.decoder_seq_len * self.features_n,
			bias = False
		)
		init.normal_(self.exog_decode_out_fc.weight)
		
		# 内生变量.
		self.endo_decode_out_fc = nn.Linear(
			self.decoder_seq_len * (self.exog_seq2seq_params['hidden_size'] + self.endo_seq2seq_params['hidden_size']),
			self.endo_nodes_n * self.decoder_seq_len * self.features_n,
			bias = False
		)
		init.normal_(self.endo_decode_out_fc.weight)
		
	def forward(self, x: torch.Tensor):
		# x.shape = (batch_size, nodes_n, seq_len, features_n)
		batch_size_ = x.shape[0]
		
		# 计算GCN layer输出.
		gcn_layer_out_ = self.gcn_layer(x)  # gcn_layer_out_.shape = (seq_len, nodes_n, batch_size, hidden_size)
		
		# 计算exog_seq2seq输出.
		gcn_layer_out_ = gcn_layer_out_.transpose(0, 2)  # shape = (batch_size, nodes_n, seq_len, hidden_size)
		gcn_layer_out_ = gcn_layer_out_.transpose(1, 2)  # shape = (batch_size, seq_len, nodes_n, hidden_size)
		
		# gcn_layer_out_.shape = (batch_size, seq_len, nodes_n * gcn_hidden_size)
		gcn_layer_out_ = gcn_layer_out_.reshape(batch_size_, self.encoder_seq_len, self.nodes_n * self.gcn_params['hidden_size'])
		
		# 拆分为内生变量和外生变量类型.
		endo_encoder_in_ = gcn_layer_out_[:, :, :self.gcn_params['endo_exog_split_loc']]  # shape = (batch_size, seq_len, endo_nodes_n * hidden_size)
		exog_encoder_in_ = gcn_layer_out_[:, :, self.gcn_params['endo_exog_split_loc']:]  # shape = (batch_size, seq_len, exog_nodes_n * hidden_size)
		
		# 计算exog_seq2seq输出.
		# exog_decoder_out_ = (batch_size, decoder_seq_len, exog_hidden_size)
		exog_encoder_out_, exog_decoder_out_ = self.exog_seq2seq(exog_encoder_in_)
		
		# 计算endo_seq2seq输出.
		# endo_decoder_out_ = (batch_size, decoder_seq_len, endo_hidden_size)
		_, endo_decoder_out_ = self.endo_seq2seq(endo_encoder_in_, exog_encoder_out_, exog_decoder_out_)
		
		# 输出结果经过FC变换.
		endo_decoder_out_ = endo_decoder_out_.reshape(batch_size_, self.decoder_seq_len * self.endo_seq2seq_params['hidden_size'])
		exog_decoder_out_ = exog_decoder_out_.reshape(batch_size_, self.decoder_seq_len * self.exog_seq2seq_params['hidden_size'])
		
		exog_fc_out_ = self.exog_decode_out_fc(exog_decoder_out_)  # shape = (batch_size, nodes_n, seq_len, features_n)
		endo_fc_out_ = self.endo_decode_out_fc(
			torch.cat((endo_decoder_out_, exog_decoder_out_), dim = 1))  # shape = (batch_size, nodes_n, seq_len, features_n)
		
		# %% 激活函数.
		fc_out_ = f.softplus(torch.cat((endo_fc_out_, exog_fc_out_), dim = 1))
		fc_out_ = fc_out_.reshape(batch_size_, self.nodes_n,
		                          self.decoder_seq_len * self.features_n)  # shape = (batch_size, nodes_n, seq_len * features_n)
		fc_out_ = fc_out_.transpose(1, 2)  # shape = (batch_size, seq_len * featrues_n, nodes_n)
		
		y = fc_out_
		return y
		


if __name__ == '__main__':
	# %% 测试参数.
	from lib.step_2_train_model.tmp import FEATURES_N, ENDO_NODES_N, EXOG_NODES_N, L
	from lib.step_2_train_model.tmp import hist_len, pred_len
	from lib.step_2_train_model.tmp import trainloader, verifloader

	x_train, y_train = [p for p in trainloader][0]

	# x.shape = (batch_size, nodes_n, seq_len, features_n)
	# y.shape = (batch_size, pred_len * nodes_n * features_n)
	x, y = x_train, y_train
	encoder_seq_len, decoder_seq_len = hist_len + 1, pred_len

	# %% 测试GCN + Seq2Seq.
	self = GCNSeq2Seq(
		endo_nodes_n = ENDO_NODES_N,
		exog_nodes_n = EXOG_NODES_N,
		features_n = FEATURES_N,
		L = L,
		encoder_seq_len = encoder_seq_len,
		decoder_seq_len = decoder_seq_len
	)

	gcn_layer_params = {
		'hidden_size': 8,
		'layers_n': 2
	}
	self.init_gcn_layer(**gcn_layer_params)

	exog_seq2seq_params = {
		'hidden_size': 8,
		'layers_n': 3,
		'dropout': 0.0
	}
	self.init_exog_seq2seq(**exog_seq2seq_params)

	endo_seq2seq_params = {
		'hidden_size': 8,
		'layers_n': 3,
		'dropout': 0.0
	}
	self.init_endo_seq2seq(**endo_seq2seq_params)

	# %% 前向计算.
	# x.shape = (batch_size, nodes_n, seq_len, features_n)
	batch_size_ = x.shape[0]

	# 计算GCN layer输出.
	gcn_layer_out_ = self.gcn_layer(x)  # gcn_layer_out_.shape = (seq_len, nodes_n, batch_size, hidden_size)

	# 计算exog_seq2seq输出.
	gcn_layer_out_ = gcn_layer_out_.transpose(0, 2)  # shape = (batch_size, nodes_n, seq_len, hidden_size)
	gcn_layer_out_ = gcn_layer_out_.transpose(1, 2)  # shape = (batch_size, seq_len, nodes_n, hidden_size)

	# gcn_layer_out_.shape = (batch_size, seq_len, nodes_n * gcn_hidden_size)
	gcn_layer_out_ = gcn_layer_out_.reshape(batch_size_, self.encoder_seq_len, self.nodes_n * self.gcn_params['hidden_size'])

	# 拆分为内生变量和外生变量类型.
	endo_encoder_in_ = gcn_layer_out_[:, :, :self.gcn_params['endo_exog_split_loc']]  # shape = (batch_size, seq_len, endo_nodes_n * hidden_size)
	exog_encoder_in_ = gcn_layer_out_[:, :, self.gcn_params['endo_exog_split_loc']:]  # shape = (batch_size, seq_len, exog_nodes_n * hidden_size)

	# 计算exog_seq2seq输出.
	# exog_decoder_out_ = (batch_size, decoder_seq_len, exog_hidden_size)
	exog_encoder_out_, exog_decoder_out_ = self.exog_seq2seq(exog_encoder_in_)

	# 计算endo_seq2seq输出.
	# endo_decoder_out_ = (batch_size, decoder_seq_len, endo_hidden_size)
	_, endo_decoder_out_ = self.endo_seq2seq(endo_encoder_in_, exog_encoder_out_, exog_decoder_out_)

	# 输出结果经过FC变换.
	endo_decoder_out_ = endo_decoder_out_.reshape(batch_size_, self.decoder_seq_len * self.endo_seq2seq_params['hidden_size'])
	exog_decoder_out_ = exog_decoder_out_.reshape(batch_size_, self.decoder_seq_len * self.exog_seq2seq_params['hidden_size'])

	exog_fc_out_ = self.exog_decode_out_fc(exog_decoder_out_)  # shape = (batch_size, nodes_n, seq_len, features_n)
	endo_fc_out_ = self.endo_decode_out_fc(
		torch.cat((endo_decoder_out_, exog_decoder_out_), dim = 1))  # shape = (batch_size, nodes_n, seq_len, features_n)

	# %% 激活函数.
	fc_out_ = f.softplus(torch.cat((endo_fc_out_, exog_fc_out_), dim = 1))
	fc_out_ = fc_out_.reshape(batch_size_, self.nodes_n, self.decoder_seq_len * self.features_n)  # shape = (batch_size, nodes_n, seq_len * features_n)
	fc_out_ = fc_out_.transpose(1, 2)  # shape = (batch_size, seq_len, nodes_n)
