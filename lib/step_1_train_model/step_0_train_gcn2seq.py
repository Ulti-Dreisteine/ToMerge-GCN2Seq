# -*- coding: utf-8 -*-
"""
Created on 2020/4/25 17:42

@Project -> File: gcn -> step_0_train_gcn_lstm.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: train a GCN2Seq model
"""

import logging

logging.basicConfig(level = logging.INFO)

import matplotlib.pyplot as plt
import numpy as np
import torch
import sys, os

sys.path.append('../..')

from lib import proj_dir
from lib import COLS_BOUNDS
from mod.deep_learning_relevant.gcn2seq import GCN2Seq
from mod.deep_learning_relevant.loss_criterion import criterion
from mod.deep_learning_relevant.read_n_save_model import save_model

# ============ Whether use GPU or not ============
use_cuda = torch.cuda.is_available()
print('CUDA: {}'.format(use_cuda))

# ============ Load test data & params ============
from lib.step_1_train_model import COLS, NODES_N, ENDO_NODES_N, EXOG_NODES_N, FEATURES_N, L
from lib.step_1_train_model import hist_len, pred_len, samples_n, train_ratio, batch_size
from lib.step_1_train_model import build_total_samples_tensors, build_train_verif_loader, gen_laplacian_matrix

x, y = build_total_samples_tensors(samples_n, hist_len, pred_len, NODES_N, FEATURES_N)

# ============ Build train & verify data set ============
trainloader, verifloader = build_train_verif_loader(x, y, train_ratio, batch_size)

# ============ Initialize the GCN2Seq model, some params require manually input ============
encoder_seq_len = hist_len + 1
decoder_seq_len = pred_len
self = GCN2Seq(encoder_seq_len, decoder_seq_len)

gcn_params = {
	'endo_nodes_n': ENDO_NODES_N,
	'exog_nodes_n': EXOG_NODES_N,
	'features_n': FEATURES_N,
	'L': L,
	'hidden_size': 6,
	'layers_n': 2}
lstm_params = {
	'hidden_size': 8,
	'layers_n': 3,
	'bias': False,
	'batch_first': True,
	'dropout': 0.0
}

self.init_gcn_layer(**gcn_params)
self.init_encoders_n_decoders(**lstm_params)

if use_cuda:
	torch.cuda.empty_cache()
	trainloader = [(train_x.cuda(), train_y.cuda()) for (train_x, train_y) in trainloader]
	verifloader = [(verify_x.cuda(), verify_y.cuda()) for (verify_x, verify_y) in verifloader]
	self = self.cuda()

# ============ Train the model ============
model_struc_params = {
	'encoder_seq_len': self.encoder_seq_len,
	'decoder_seq_len': self.decoder_seq_len,
	'gcn_params': {
		'endo_nodes_n': self.gcn_params['endo_nodes_n'],
		'exog_nodes_n': self.gcn_params['exog_nodes_n'],
		'features_n': self.gcn_params['features_n'],
		'hidden_size': self.gcn_params['hidden_size'],
		'layers_n': self.gcn_params['layers_n']
	},
	'lstm_params': {
		'hidden_size': self.lstm_params['hidden_size'],
		'layers_n': self.lstm_params['layers_n'],
		'bias': self.lstm_params['bias'],
		'batch_first': self.lstm_params['batch_first'],
		'dropout': self.lstm_params['dropout'],
	}
}

if 'model' not in os.listdir(os.path.join(proj_dir, 'file')):
	os.mkdir(os.path.join(proj_dir, 'file/model/'))
model_fp = os.path.join(proj_dir, 'file/model/')

if 'gcn_seq2seq' not in os.listdir(model_fp):
	os.mkdir(os.path.join(model_fp, 'gcn_seq2seq'))
model_save_fp = os.path.join(proj_dir, 'file/model/gcn_seq2seq/')

lr = 0.01
optimizer = torch.optim.Adam(self.parameters(), lr = lr)
train_loss_record, verif_loss_record = None, None
for epoch in range(1000):
	self.train()
	s = 0
	for x_train, y_train in trainloader:
		# x.shape = (batch_size, nodes_n, seq_len, features_n)
		y_pred = self(x_train)  # shape = (batch_size, pred_seq_len, nodes_n)

		# 计算loss.
		y_pred = y_pred.reshape(batch_size, self.decoder_seq_len, self.gcn_params['nodes_n'] * self.gcn_params['features_n'])
		train_loss = criterion(y_train.flatten(), y_pred.flatten())

		optimizer.zero_grad()
		train_loss.backward()
		optimizer.step()

		if s == 0:
			train_loss = train_loss.reshape(1).detach()
			if epoch == 0:
				train_loss_record = train_loss
			else:
				train_loss_record = torch.cat((train_loss_record, train_loss), dim = 0)

		s += 1

	self.eval()
	with torch.no_grad():
		s = 0
		for x_verif, y_verif in verifloader:
			y_pred = self(x_verif)  # shape = (batch_size, pred_seq_len, nodes_n)

			# 计算loss.
			y_pred = y_pred.reshape(batch_size, self.decoder_seq_len, self.gcn_params['nodes_n'] * self.gcn_params['features_n'])
			verif_loss = criterion(y_verif.flatten(), y_pred.flatten())

			if s == 0:
				verif_loss = verif_loss.reshape(1).detach()
				if epoch == 0:
					verif_loss_record = verif_loss
				else:
					verif_loss_record = torch.cat((verif_loss_record, verif_loss), dim = 0)

			s += 1

			break

	if (epoch + 1) % 5 == 0:
		print(epoch, train_loss_record[-1], verif_loss_record[-1])

	if (epoch + 1) % 10 == 0:
		save_model(self, model_struc_params, train_loss_record, verif_loss_record, model_save_fp)

save_model(self, model_struc_params, train_loss_record, verif_loss_record, model_save_fp)

# ============ Draw the loss curves ============
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(train_loss_record.detach().cpu().numpy().flatten()[100:])
plt.subplot(2, 1, 2)
plt.plot(verif_loss_record.detach().cpu().numpy().flatten()[100:])

# ============ Verify the model effect ============
self.eval()
x_verify, y_verif = verifloader[0]
y_pred = self(x_verify)
y_pred = y_pred.reshape(batch_size, self.decoder_seq_len, self.gcn_params['nodes_n'] * self.gcn_params['features_n'])
y_verif = y_verif.reshape(batch_size, self.decoder_seq_len, self.gcn_params['nodes_n'] * self.gcn_params['features_n'])

y_verif = y_verif.detach().cpu().numpy()
y_pred = y_pred.detach().cpu().numpy()


def mae_func(y_true, y_pred):
	return np.mean(np.abs(y_true - y_pred))


plt.figure(figsize = [6, 20])
nodes = COLS
for p in range(len(COLS)):
	mae = []
	for i in range(decoder_seq_len):
		mae.append(mae_func(y_verif[:, i, p], y_pred[:, i, p]))
	mae = np.array(mae)
	plt.subplot(len(COLS), 1, p + 1)
	plt.plot(mae * COLS_BOUNDS[nodes[p]][1])
	plt.legend([nodes[p]], loc = 'upper right')
	plt.grid(True)
	
	if p == len(COLS) - 1:
		plt.xlabel('pred time step (hr)')
plt.tight_layout()
