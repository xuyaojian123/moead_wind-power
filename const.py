#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : const.py
@Author: XuYaoJian
@Date  : 2022/7/24 21:35
@Desc  : 
"""


# k                 pop[i][0]     2,3,4,5
# seq_len           pop[i][1]     6,7,8,9,10,11,12,13
# batch_size        pop[i][2]     32,64,128,256
# learning_rate     pop[i][3]     0.005,0.01,0.05,0.1,0.5
# optimize          pop[i][4]     Adam，SGD，Adamax
# num_layer         pop[i][5]     5 1,2,3
# num_layer_unit    pop[i][6]     30,31,32,33,34,35,36,37,38,39,40
# dropout           pop[i][7]     0,0.25,0.5
# dense             pop[i][8]     1,2,3
# dense_unit        pop[i][9]     10,15,20,25,30

class Parameters(object):
    # k = [2, 3, 4, 5]
    k = [2, 3]
    # seq_len = [6, 7, 8, 9, 10, 11, 12, 13]
    seq_len = [7, 8, 9, 10, 11, 12, 13, 14]
    # batch_size = [32, 64, 128, 256]
    batch_size = [128, 256]
    # learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    learning_rate = [0.001, 0.005, 0.01, 0.1]
    # optimizer = [0, 1, 2]
    optimizer = [0, 1]
    # num_layer = [1, 2, 3]
    num_layer = [1, 2]
    # hidden_size = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    hidden_size = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    # dropout = [0, 0.25, 0.5]
    dropout = [0]
    # dense = [1, 2, 3, 4]
    dense = [1, 2, 3, 4]
    # dense_unit = [10, 15, 20, 25, 30]
    dense_unit = [20, 30]

# print(Parameters.k)
