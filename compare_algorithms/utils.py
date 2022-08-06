#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : utils.py
@Author: XuYaoJian
@Date  : 2022/8/1 20:11
@Desc  : 
"""
import pandas as pd
import numpy as np


# convert an array of values into a dataset matrix
def create_dataset(filename, seq_len=9):
    df = pd.read_csv(filename)
    data = df['wind speed at 100m (m/s)']
    result = []
    for index in range(len(data) - seq_len):
        result.append(data[index:index + seq_len + 1])  # create train label
    # 划分训练集和测试集 (测试集为两天，风能间隔为10min,两天=6*24*2=288)
    result = np.array(result)  # 用numpy对其进行矩阵化
    row = int(result.shape[0] * (1 - 288 / result.shape[0]))
    x_train = result[:row, :-1]
    y_train = result[:row, -1]
    x_test = result[row:, :-1]
    y_test = result[row:, -1]
    return x_train, y_train, x_test, y_test
