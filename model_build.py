#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : model_build.py
@Author: XuYaoJian
@Date  : 2022/7/5 10:37
@Desc  : 根据参数定义网络模型 GRU + Linear
"""

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, SGD, Adamax


def model_build(individual):
    # 建立模型
    seq_len = int(individual[1])
    learning_rate = individual[3]
    # 优化器
    optimizer = int(individual[4])
    if optimizer == 0:
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == 1:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    elif optimizer == 2:
        optimizer = Adamax(learning_rate=learning_rate)
    num_layers = int(individual[5])
    hidden_size = int(individual[6])
    dropout = individual[7]
    dense = int(individual[8])
    dense_units = int(individual[9])
    model = Sequential()
    model.add(Input(shape=(seq_len, 1)))
    for i in range(num_layers - 1):
        # model.add(GRU(units=hidden_size, return_sequences=True, kernel_initializer='glorot_normal', dropout=dropout))
        model.add(GRU(units=hidden_size, return_sequences=True))
    # model.add(GRU(units=hidden_size, return_sequences=False, kernel_initializer='glorot_normal', dropout=dropout))
    model.add(GRU(units=hidden_size, return_sequences=False))
    units = dense_units
    for i in range(dense):
        # model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
        model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
        units = int(units / 2)
    # model.add(Dense(1, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation='relu', kernel_initializer='he_normal'))
    # loss_fn = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
    # model.compile(optimizer=optimizer, loss='mse')
    model.compile(optimizer=optimizer, loss='mse')
    return model
