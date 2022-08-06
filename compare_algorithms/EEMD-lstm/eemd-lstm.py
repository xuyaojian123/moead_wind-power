#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : eemd-lstm.py
@Author: XuYaoJian
@Date  : 2022/8/5 14:18
@Desc  : 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,LSTM
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, SGD

from utils import load_data_with_eemd



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
        # print("Training on GPU...")
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # print("Training on GPU...")
        # for gpu in gpus:
        #     tf.config.experimental.set_memory_growth(gpu, True)


set_tf_device('gpu')


def train():
    title = "washingtong_autumn_20121001-20121007"
    filename = '../../data/week_data/data/washingtong_autumn_20121001-20121007新.csv'
    K = 3
    seq_len = 9
    x_trains, y_trains, x_tests, y_tests = load_data_with_eemd(filename, K, seq_len)
    x_trains, y_trains, x_tests, y_tests = np.array(x_trains), np.array(y_trains), np.array(x_tests), np.array(y_tests)
    print(x_trains.shape, y_trains.shape, x_tests.shape, y_tests.shape)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    predict_eemd_lstm = 0
    for k in range(K):
        # define model
        model = Sequential()
        model.add(Input(shape=(9, 1)))
        model.add(LSTM(32, return_sequences=False))
        model.add(Dense(16, kernel_initializer='he_normal', activation='relu'))
        model.add(Dense(8, kernel_initializer='he_normal', activation='relu'))
        model.add(Dense(4, kernel_initializer='he_normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='he_normal'))
        # 对学习率很敏感
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        model.fit(x_trains[k], y_trains[k], epochs=150, batch_size=64, validation_split=0.1, verbose=1, callbacks=[callback])
        predict_eemd_lstm += model.predict(x_tests[k])

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.set_title(title)
    ax.plot(y_tests[K], label='true')
    ax.plot(predict_eemd_lstm, label='predict')
    ax.grid()
    ax.legend()
    plt.show()

    mse = cal_mse(predict_eemd_lstm, y_tests[K])
    print(f"mse:{mse}")

    df = pd.DataFrame(data={
        'true_value': y_tests[K].reshape(-1),
        'predict': predict_eemd_lstm.reshape(-1),
    })
    df.to_csv("result/"+title+"_mse_" + str(mse) + ".csv", index=False)


def cal_mse(y_hat, y):
    mse = (np.square(y_hat - y)).mean()
    return mse


if __name__ == "__main__":
    train()
