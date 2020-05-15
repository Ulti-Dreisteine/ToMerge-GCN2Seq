# -*- coding: utf-8 -*-
"""
Created on 2020/4/30 11:47

@Project -> File: pollution-regional-forecast-offline-gcn-training -> lstm.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: Seq2Seq模型
"""

import logging

logging.basicConfig(level = logging.INFO)

from torch import nn
import torch
import sys, os

sys.path.append('../..')


class Seq2Seq(nn.Module):
	"""
	Seq2Seq时间序列模型

	Attributes:
		batch_first: bool, 是否将batch放在第一维
		bias: bool, 是否加入偏置项

	Methods:
		init_encoder: 初始化编码器
		init_decoder: 初始化解码器
		forward: 前向计算
	"""
	
	def __init__(self, batch_first: bool = True, bias: bool = False):
		super(Seq2Seq, self).__init__()
		self.batch_first = batch_first  # always batch_first = True
		self.bias = bias
	
	def init_encoder(self, input_size: int, hidden_size: int, layers_n: int, dropout: float = 0.0):
		self.encoder = nn.LSTM(
			input_size = input_size,
			hidden_size = hidden_size,
			num_layers = layers_n,
			bias = self.bias,
			batch_first = self.batch_first,
			dropout = dropout
		)
	
	def init_decoder(self, input_size: int, decoder_seq_len: int, dropout: float = 0.0):
		"""
		解码器
		:param input_size: int, 输入维数, 一般没有其他接入时为encoder.hidden_size
		:param decoder_seq_len: int, 解码seq_len
		:param dropout: float, 随机dropout比例
		"""
		self.decoder = nn.LSTM(
			input_size = input_size,
			hidden_size = self.encoder.hidden_size,
			num_layers = self.encoder.num_layers,
			bias = self.bias,
			batch_first = self.batch_first,
			dropout = dropout
		)
		self.decoder_seq_len = decoder_seq_len
	
	def forward(self, x: torch.Tensor, external_encoder_input: torch.Tensor = None, external_decoder_input: torch.Tensor = None) -> (torch.Tensor,
	                                                                                                                                 torch.Tensor):
		"""
		:param external_encoder_input: torch.Tensor, 外部编码输入, shape = (batch_size, encoder_seq_len, hidden_size)
		:param external_decoder_input: torch.Tensor, 外部解码输入, shape = (batch_size, decoder_seq_len, hidden_size)
		:param x: torch.Tensor, shape = (batch_size, encoder_seq_len, input_size)
		:return: decoder_out: torch.Tensor, shape = (batch_size, decoder_seq_len, hidden_size)
		"""
		# 编码.
		# encoder_out_.shape = (batch_size, seq_len, hidden_size)
		# h_encoder_.shape = c_encoder_.shape = (layers_n, batch_size, hidden_size)
		if external_encoder_input is not None:
			x = torch.cat((x, external_encoder_input), dim = 2)
		encoder_out, (h_encoder_out_, c_encoder_out_) = self.encoder(x)
		
		# 循环解码.
		decoder_out = None
		y_ = encoder_out[:, -1:, :]
		h_, c_ = h_encoder_out_, c_encoder_out_
		for i in range(self.decoder_seq_len):
			
			if external_decoder_input is not None:
				y_ = torch.cat((y_, external_decoder_input[:, i: i + 1, :]), dim = 2)
				
			y_, (h_, c_) = self.decoder(y_, (h_, c_))
			if i == 0:
				decoder_out = y_
			else:
				decoder_out = torch.cat((decoder_out, y_), dim = 1)
		
		return encoder_out, decoder_out
	

if __name__ == '__main__':
	pass



