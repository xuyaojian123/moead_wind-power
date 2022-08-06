#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : dataset.py
@Author: XuYaoJian
@Date  : 2022/7/13 19:07
@Desc  :
"""
from typing import Any

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class PowerDataset(Dataset):
    # 初始化函数，传入数据
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        x = self.X[index]
        y = self.Y[index]
        return x, y

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.X)
