#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : mlp.py
@Author: XuYaoJian
@Date  : 2022/8/2 14:46
@Desc  : 
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
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
    model = Sequential()
    model.add(Input(shape=(9,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1, activation='relu'))
    # compile the model
    # 对学习率很敏感
    opt = Adam(learning_rate=0.02)
    model.compile(optimizer=opt, loss='mse')
    title = "California_autumn_20121001-20121007"
    filename = '../../data/week_data/data/California_autumn_20121001-20121007新.csv'
    x_train, y_train, x_test, y_test = create_dataset(filename, seq_len=9)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=8, restore_best_weights=True)
    # fit the model
    # model.fit(x_train, y_train, epochs=200, batch_size=64,validation_split=0.1,verbose=1,callbacks=[callback])
    model.fit(x_train, y_train, epochs=200, validation_split=0.1, batch_size=64, verbose=1)
    yhat = model.predict(x_test)
    print(yhat.shape)

    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    ax.set_title(title)
    ax.plot(y_test, label='true')
    ax.plot(yhat, label='predict')
    ax.grid()
    ax.legend()
    plt.show()

    yhat = yhat.reshape(-1)
    mse = cal_mse(yhat, y_test)
    print(f"mse:{mse}")

    df = pd.DataFrame(data={
        'true_value': y_test,
        'predict': yhat
    })
    df.to_csv("result/"+title+"_mse_" + str(mse) + ".csv", index=False)


def cal_mse(y_hat, y):
    mse = (np.square(y_hat - y)).mean()
    return mse


if __name__ == "__main__":
    train()
