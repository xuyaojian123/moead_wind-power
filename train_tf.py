#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : train.py
@Author: XuYaoJian
@Date  : 2022/7/13 16:25
@Desc  :
"""
import os
import random
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
# import tensorflow._api.v2.compat.v1 as tf
from keras import backend as K
from tensorflow.python.framework import ops

from utils import load_data_with_vmd, maponezero
from model_build import model_build


def setup_seed(seed):
    # random.Random(seed).shuffle(arr)
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(42)


def set_tf_device(device):
    if device == 'cpu':
        print("Training on CPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == 'gpu':
        gpus = tf.config.experimental.list_physical_devices("GPU")
        # 前面一直用的第一张卡跑，一次性跑不了这么多个程序，换到第三张卡跑
        # os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        # tf.config.experimental.set_memory_growth(gpu[0], True)
        # tf.config.set_visible_devices(gpus[1], 'GPU')
        print("Training on GPU...")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        print("Training on GPU...")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


set_tf_device('gpu')

individual_index = 0


def train(moead, individual):
    global individual_index
    print("第%d评价个体：\n" % (individual_index + 1))
    individual_index += 1
    # batch_size
    batch_size = int(individual[2])
    # 分解个数k
    vmd_k = int(individual[0])
    # 序列步长
    seq_len = int(individual[1])
    [X_trains, Y_trains, X_tests, Y_tests] = load_data_with_vmd(moead.filename, vmd_k, seq_len)
    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    x_maxmins = []
    y_maxmins = []
    for k in range(vmd_k):
        # normalize data
        [x_train, x_maxmin] = maponezero(X_trains[k])
        [y_train, y_maxmin] = maponezero(Y_trains[k])
        x_test = maponezero(X_tests[k], "apply", x_maxmin)
        y_test = maponezero(Y_tests[k], "apply", y_maxmin)
        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)
        x_maxmins.append(x_maxmin)
        y_maxmins.append(y_maxmin)

    # 分解成k个序列，分别有k个模型训练
    # models = []
    best_mse = 10000.
    best_std = 10000.
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    for repeat in range(moead.trainging_repeat):
        predict_test_sum = 0
        for i in range(0, vmd_k):
            model = model_build(individual)
            print("第%d个子序列\n" % (i + 1))
            # history = model.fit(x_trains[i], y_trains[i], epochs=moead.epoch, batch_size=batch_size, verbose=1)
            history = model.fit(x_trains[i], y_trains[i], epochs=moead.epoch, batch_size=batch_size, callbacks=[callback], verbose=1)
            # history = model.fit(x_trains[i], y_trains[i], epochs=moead.epoch, batch_size=batch_size,validation_split=0.1,
            #                     verbose=1)
            predict_sequence = model.predict(x_tests[i])
            predict_sequence_reverse = maponezero(predict_sequence, 'reverse', y_maxmins[i])
            predict_test_sum += predict_sequence_reverse
            # clear model data and state to avoid memory leak
            K.clear_session()
            ops.reset_default_graph()
            # tf.reset_default_graph()

        mse, std = cal_mse(predict_test_sum, Y_tests[vmd_k]), cal_std(predict_test_sum, Y_tests[vmd_k])
        if mse < best_mse:
            best_mse = mse
            best_std = std

    return best_mse, best_std


def cal_std(y_hat, y):
    error = y_hat - y
    std = np.std(error)
    return std


def cal_mse(y_hat, y):
    mse = (np.square(y_hat - y)).mean()
    return mse
