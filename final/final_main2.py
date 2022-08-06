#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : final_main1.py
@Author: XuYaoJian
@Date  : 2022/7/30 16:28
@Desc  : 
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from tensorflow.python.framework import ops

from utils import load_data_with_vmd, maponezero
from model_build import model_build

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt


def set_tf_device(device):
    if device == 'cpu':
        print("Training on CPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == 'gpu':
        gpus = tf.config.experimental.list_physical_devices("GPU")
        # 前面一直用的第一张卡跑，一次性跑不了这么多个程序，换到第三张卡跑
        tf.config.set_visible_devices(gpus[3], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[3], True)

        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # print("Training on GPU...")
        # for gpu in tf.config.experimental.list_physical_devices("GPU"):
        #     tf.config.experimental.set_memory_growth(gpu, True)


set_tf_device('gpu')


# 从最后一次的迭代种群里面，挑选mse最小的个体
def create_individual():
    individual = []
    k = 3
    seq_len = 9
    batch_size = 64
    learning_rate = 0.01
    optimizer = 0
    num_layer = 1
    hidden_size = 32
    dropout = 0
    dense = 3
    dense_unit = 16
    individual.extend(
        [k, seq_len, batch_size, learning_rate, optimizer, num_layer, hidden_size, dropout, dense, dense_unit])
    return individual


def train(repeat, epoch, filename, save_filename, individual):
    # batch_size
    batch_size = int(individual[2])
    # 分解个数k
    vmd_k = int(individual[0])
    # 序列步长
    seq_len = int(individual[1])
    [x_trains, y_trains, x_tests, y_tests] = load_data_with_vmd(filename, vmd_k, seq_len)
    x_trains, y_trains, x_tests, y_tests = np.array(x_trains), np.array(y_trains), np.array(x_tests), np.array(y_tests)
    print(x_trains.shape, y_trains.shape, x_tests.shape, y_tests.shape)

    # 分解成k个序列，分别有k个模型训练
    best_mse = 10000.
    best_std = 10000.
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    for r in range(repeat):
        predict_test_sum = 0
        for i in range(0, vmd_k):
            # define model
            model = Sequential()
            model.add(Input(shape=(9, 1)))
            model.add(GRU(32, return_sequences=False))
            model.add(Dense(16, kernel_initializer='he_normal', activation='relu'))
            model.add(Dense(8, kernel_initializer='he_normal', activation='relu'))
            model.add(Dense(4, kernel_initializer='he_normal', activation='relu'))
            model.add(Dense(1, kernel_initializer='he_normal'))
            # 对学习率很敏感
            model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
            model.fit(x_trains[i], y_trains[i], epochs=200, batch_size=64, validation_split=0.1, verbose=1,
                      callbacks=[callback])
            predict_test_sum += model.predict(x_tests[i])

            # model = model_build(individual)
            # print("第%d个子序列\n" % (i + 1))
            # # history = model.fit(x_trains[i], y_trains[i], epochs=epoch, batch_size=batch_size, callbacks=[callback], verbose=1,validation_split=0.1)
            # history = model.fit(x_trains[i], y_trains[i], epochs=epoch, batch_size=batch_size, callbacks=[callback],
            #                     verbose=1, validation_split=0.1)
            #
            # predict_sequence = model.predict(x_tests[i])
            # # predict_sequence_reverse = maponezero(predict_sequence, 'reverse', y_maxmins[i])
            # predict_test_sum += predict_sequence
            # # clear model data and state to avoid memory leak
            # # K.clear_session()
            # # ops.reset_default_graph()
            # # tf.reset_default_graph()

        mse, std = cal_mse(predict_test_sum, y_tests[vmd_k]), cal_std(predict_test_sum, y_tests[vmd_k])
        if mse < best_mse:
            df = pd.DataFrame(data={
                'true_value': y_tests[vmd_k].reshape(-1),
                'predict': predict_test_sum.reshape(-1)
            })
            df.to_csv("{}_mse{:.4f}_std{:.4f}.csv".format(save_filename, mse, std), index=False)
            best_mse = mse
            best_std = std

            fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
            ax.set_title("washingtong_summer_20120701-20120707")
            ax.plot(y_tests[vmd_k], label='true')
            ax.plot(predict_test_sum, label='predict')
            ax.grid()
            ax.legend()
            plt.show()

    return best_mse, best_std


def cal_std(y_hat, y):
    error = y_hat - y
    std = np.std(error)
    return std


def cal_mse(y_hat, y):
    mse = (np.square(y_hat - y)).mean()
    return mse


if __name__ == '__main__':
    individual = create_individual()
    repeat = 1
    epoch = 200
    filename = '../data/week_data/data/washingtong_summer_20120701-20120707新.csv'
    save_filename = '../result/washingtong_summer_20120701-20120707/predict/result'
    mse, std = train(repeat, epoch, filename, save_filename, individual)
    print("mse:{:.4f}, std:{:.4f}".format(mse, std))
