# -*- coding: utf-8 -*-
"""
Created on 2019/12/9 下午3:56

@Project -> File: industrial-research-guodian-project -> time_conversion.py

@Author: luolei

@Email: dreisteine262@163.com

@Describe: 时间转换
"""

import time

t0 = (1970, 1, 1, 8, 0, 0, 3, 1, 0)


def time2stp(t, time_format = '%Y-%m-%dT%H:%M:%S+08:00'):
	"""时间转为时间戳"""
	stp = int(time.mktime(time.strptime(t, time_format)) - time.mktime(t0))
	return stp


def stp2time(stp, time_format = '%Y-%m-%dT%H:%M:%S+08:00'):
	"""时间戳转换为时间"""
	t_arr = time.localtime(stp + int(time.mktime(t0)))
	t = time.strftime(time_format, t_arr)
	return t


def time2utc(t):
	"""
	将北京时间转换为UTC时间
	:param t: str like "%Y-%m-%d %H:%M:%S", 当地时间
	:return: t_utc: str like "%Y-%m-%dT%H:%M:%S+08:00", 转换后的UTC时间
	"""
	
	t_date = time.strptime(t, '%Y-%m-%d %H:%M:%S')
	stp = int(time.mktime(t_date) - 8 * 3600)  # **注意时区
	t_utc = time.strftime('%Y-%m-%dT%H:%M:%S+08:00', time.localtime(stp))
	
	return t_utc


def utc2stp(utc):
	"""
	将UTC时间转为时间戳
	:param utc: str like "%Y-%m-%dT%H:%M:%S+08:00", UTC时间
	:return: stp: int, 转换后的时间戳
	"""
	t_date = time.strptime(utc, '%Y-%m-%dT%H:%M:%S+08:00')
	t_start = (1970, 1, 1, 8, 0, 0, 3, 1, 0)  # 时间戳的起始记录时间，防止因时区不同导致计算出错
	stp = int(time.mktime(t_date) - time.mktime(t_start) + 8 * 3600)
	
	return stp


if __name__ == '__main__':
	t = '2019-06-19T16:00:00+08:00'
	stp = time2stp(t)
	t = stp2time(stp)



