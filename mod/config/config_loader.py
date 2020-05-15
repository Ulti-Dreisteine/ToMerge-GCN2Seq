# -*- coding: utf-8 -*-
"""
Created on 2019/11/22 上午9:24

@Project -> File: ode-neural-network -> config_loader.py

@Author: luolei

@Describe: 项目参数配置器
"""

import lake.decorator
import lake.dir
import logging
import logging.config
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../config/'))

if len(logging.getLogger().handlers) == 0:
	logging.basicConfig(level = logging.DEBUG)


@lake.decorator.singleton
class ConfigLoader(object):
	def __init__(self):
		self._set_proj_dir()
		self._set_proj_cmap()
		self._load_config()
		self._load_env_config()
		self._load_test_config()
	
	def _set_proj_dir(self):
		"""项目根目录"""
		self._proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
	
	def _set_proj_cmap(self):
		"""设置项目颜色方案"""
		self._proj_cmap = {
			'blue': '#1f77b4',  # 蓝色
			'orange': '#ff7f0e',  # 黄橙色
			'green': '#2ca02c',  # 绿色
			'red': '#d62728',  # 红色
			'purple': '#9467bd',  # 紫色
			'cyan': '#17becf',  # 青色
			'grey': '#7f7f7f',  # 灰色
			'black': 'k'  # 黑色
		}
	
	def _load_config(self):
		"""载入config.yml中的参数配置"""
		self._config_path = os.path.join(self.proj_dir, 'config/config.yml')
		with open(self._config_path, 'r', encoding = 'utf-8') as f:
			self._conf = yaml.load(f, Loader = yaml.Loader)  # yaml.FullLoader
	
	def _load_env_config(self):
		"""载入环境变量配置"""
		config_dir_ = os.path.join(self.proj_dir, 'config/')
		
		# 如果本地config中有master.yml则优先使用master, 否则使用default.yml, 否则为空字典.
		if 'master.yml' in os.listdir(config_dir_):
			print('Use env variables in master.yml.')
			env_config_path_ = os.path.join(config_dir_, 'master.yml')
		else:
			if 'default.yml' in os.listdir(config_dir_):
				print('Use env variables in default.yml.')
				env_config_path_ = os.path.join(config_dir_, 'default.yml')
			else:
				env_config_path_ = None
		
		if env_config_path_ is None:
			self._local_env_conf = {}
		else:
			with open(env_config_path_, 'r', encoding = 'utf-8') as f:
				self._local_env_conf = yaml.load(f, Loader = yaml.Loader)
		
		# 线上环境变量注入.
		# 如果存在可注入环境变量, 则采用注入值, 否则采用环境变量配置文件中的值.
		self._env_conf = {}
		if self._local_env_conf is None:
			pass
		else:
			for key in self._local_env_conf.keys():
				if key in os.environ:
					self._env_conf.update({key: os.environ[key]})
				else:
					self._env_conf.update({key: self._local_env_conf[key]})
	
	def _load_test_config(self):
		"""载入测试参数配置"""
		config_dir_ = os.path.join(self.proj_dir, 'config/')
		test_config_path_ = os.path.join(config_dir_, 'test.yml')
		with open(test_config_path_, 'r', encoding = 'utf-8') as f:
			self._test_params = yaml.load(f, Loader = yaml.Loader)
	
	@property
	def proj_dir(self):
		return self._proj_dir
	
	@property
	def proj_cmap(self):
		return self._proj_cmap
	
	@property
	def conf(self):
		return self._conf
	
	@property
	def env_conf(self):
		return self._env_conf
	
	@property
	def test_params(self):
		return self._test_params
	
	def set_logging(self):
		"""配制logging文件"""
		if 'logs' not in os.listdir(self.proj_dir):
			os.mkdir(os.path.join(self.proj_dir, 'logs/'))
		log_config = self.conf['logging']
		# update_filename(log_config)
		logging.config.dictConfig(log_config)


def update_filename(log_config: dict):
	"""
	更新logging中filename的配置
	:param log_config: dict, 日志配置
	"""
	to_log_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), '../', x))
	if 'filename' in log_config:
		log_config['filename'] = to_log_path(log_config['filename'])
	for key, value in log_config.items():
		if isinstance(value, dict):
			update_filename(value)


config = ConfigLoader()
