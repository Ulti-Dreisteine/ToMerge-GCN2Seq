# -*- coding: utf-8 -*-
"""
Created on 2020/4/30 13:47

@Project -> File: pollution-regional-forecast-offline-gcn-training -> model_op.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

import torch
import json
import sys, os

sys.path.append('../..')


def save_model(model: torch.nn.Module, model_struc_params: dict, train_loss_record: torch.Tensor, verif_loss_record: torch.Tensor, fp: str):
	"""
	保存模型文件
	:param model: torch.nn.Module, 模型module对象
	:param model_struc_params: dict, 模型结构参数
	:param train_loss_record: torch.Tensor, 模型训练loss记录
	:param verif_loss_record: torch.Tensor, 模型验证loss记录
	:param fp: str, 文件保存位置
	"""
	# 保存模型参数.
	torch.save(model.state_dict(), os.path.join(fp, 'model_state_dict.pth'))
	
	# 保存模型结构参数.
	with open(os.path.join(fp, 'model_struc_params.json'), 'w') as f:
		json.dump(model_struc_params, f)
	
	# 损失函数记录.
	train_loss_list = list(train_loss_record.detach().cpu().numpy())
	verif_loss_list = list(verif_loss_record.detach().cpu().numpy())
	train_loss_list = [float(p) for p in train_loss_list]
	verif_loss_list = [float(p) for p in verif_loss_list]
	
	with open(os.path.join(fp, 'train_loss.json'), 'w') as f:
		json.dump(train_loss_list, f)
	with open(os.path.join(fp, 'verif_loss.json'), 'w') as f:
		json.dump(verif_loss_list, f)
		

if __name__ == '__main__':
	pass



