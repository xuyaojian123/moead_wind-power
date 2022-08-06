#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : lstm.py
@Author: XuYaoJian
@Date  : 2022/8/2 16:11
@Desc  : 
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam, SGD

from compare_algorithms.utils import create_dataset


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

    title = "California_autumn_20121001-20121007"
    filename = '../../data/week_data/data/California_autumn_20121001-20121007新.csv'
    x_train, y_train, x_test, y_test = create_dataset(filename, seq_len=9)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    # fit the model
    model.fit(x_train, y_train, epochs=200, batch_size=64, validation_split=0.1, verbose=1, callbacks=[callback])
    # model.fit(x_train, y_train, epochs=200, validation_split=0.1, batch_size=64, verbose=1)
    predict_lstm = model.predict(x_test)
    print(predict_lstm.shape)

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.set_title(title)
    ax.plot(y_test, label='true')
    ax.plot(predict_lstm, label='predict')
    ax.grid()
    ax.legend()
    plt.show()

    mse = cal_mse(predict_lstm, y_test)
    print(f"mse:{mse}")

    df = pd.DataFrame(data={
        'true_value': y_test.reshape(-1),
        'predict': predict_lstm.reshape(-1),
    })
    df.to_csv("result/"+title+"_mse_" + str(mse) + ".csv", index=False)


def cal_mse(y_hat, y):
    mse = (np.square(y_hat - y)).mean()
    return mse


if __name__ == "__main__":
    train()
