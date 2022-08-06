#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : utils.py
@Author: XuYaoJian
@Date  : 2022/7/13 16:12
@Desc  : 工具类
"""
import numpy as np
import pandas as pd
from vmdpy import VMD
import ewtpy
from PyEMD import EEMD
import matplotlib.pyplot as plt


# 获取模型的总参数和可训练参数
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_num, trainable_num


def count_params(model):
    trainableParams = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    nonTrainableParams = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    totalParams = trainableParams + nonTrainableParams
    return totalParams



# 读取文件数据
def load_data_with_eemd(filename, K, seq_len):
    """
    :param filename:
    :param K:
    :param seq_len:
    :return: 训练数据和测试数据
    """
    data = pd.read_csv(filename)['wind speed at 100m (m/s)']
    data = np.array(data).reshape(-1)
    eemd = EEMD()
    eIMF = eemd.eemd(data, max_imf=2)
    # recon = eIMF.sum(axis=0)
    # plt.plot(data)
    # plt.plot(recon)
    # plt.show()

    print('> EEMD processed finish...')
    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    for i in range(K + 1):
        result = []
        if i < K:
            seq = eIMF[i]
        else:
            seq = data
        for index in range(len(seq) - seq_len):
            result.append(seq[index:index + seq_len + 1])  # create train label
        result = np.array(result)  # 用numpy对其进行矩阵化
        # 划分训练集和测试集 (测试集为两天，风能间隔为10min,两天=6*24*2=288)
        row = int(result.shape[0] * (1 - 288 / result.shape[0]))
        x_train = result[:row, :-1]
        y_train = result[:row, -1]
        x_test = result[row:, :-1]
        y_test = result[row:, -1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)

    # plot_vmd(eIMF, data)
    return [x_trains, y_trains, x_tests, y_tests]

# 读取文件数据
def load_data_with_ewt(filename, K, seq_len):
    """
    :param filename:
    :param K:
    :param seq_len:
    :return: 练数据和测试数据
    """
    data = pd.read_csv(filename)['wind speed at 100m (m/s)']
    data = np.array(data).reshape(-1)
    ewt, mfb, boundaries = ewtpy.EWT1D(f=data, N=K, detect='locmaxmin')
    recon = ewt.sum(axis=1)
    plt.plot(data)
    plt.plot(recon)
    plt.show()

    ewt = ewt.transpose(1, 0)
    print('> EWT processed finish...')
    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    for i in range(K + 1):
        result = []
        if i < K:
            seq = ewt[i]
        else:
            seq = data
        for index in range(len(seq) - seq_len):
            result.append(seq[index:index + seq_len + 1])  # create train label
        result = np.array(result)  # 用numpy对其进行矩阵化
        # 划分训练集和测试集 (测试集为两天，风能间隔为10min,两天=6*24*2=288)
        row = int(result.shape[0] * (1 - 288 / result.shape[0]))
        x_train = result[:row, :-1]
        y_train = result[:row, -1]
        x_test = result[row:, :-1]
        y_test = result[row:, -1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)

    # plot_vmd(ewt, data)
    return [x_trains, y_trains, x_tests, y_tests]

# 读取文件数据
def load_data_with_vmd(filename, K, seq_len):
    """
    :param filename:
    :param K:
    :param seq_len:
    :return: 练数据和测试数据
    """
    data = pd.read_csv(filename)['wind speed at 100m (m/s)']
    data = np.array(data).reshape(-1)
    # some parameters for VMD
    alpha = 2000  # moderate bandwidth constraint
    DC = 0  # no DC part imposed
    init = 1  # initialize omegas uniformly
    tol = 1e-7  # tolerance 0.5
    REI = np.inf
    tauo = 0
    # 寻找最佳的tau值
    for tau in [x / 10 for x in range(0, 11)]:
        u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
        temp = np.sqrt(np.sum((np.sum(u, 0) - data) ** 2, 0) / len(data))  # 把分解的序列重构回去，和原序列作对比。保留误差最小的tau
        if temp < REI:
            REI = temp
            tauo = tau
    tau = tauo
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    print('> vmd processed finish...')
    x_trains = []
    y_trains = []
    x_tests = []
    y_tests = []
    for i in range(K + 1):
        result = []
        if i < K:
            seq = u[i]
        else:
            seq = data
        for index in range(len(seq) - seq_len):
            result.append(seq[index:index + seq_len + 1])  # create train label
        result = np.array(result)  # 用numpy对其进行矩阵化
        # 划分训练集和测试集 (测试集为两天，风能间隔为10min,两天=6*24*2=288)
        row = int(result.shape[0] * (1 - 288 / result.shape[0]))
        x_train = result[:row, :-1]
        y_train = result[:row, -1]
        x_test = result[row:, :-1]
        y_test = result[row:, -1]
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))
        y_test = np.reshape(y_test, (y_test.shape[0], 1))

        x_trains.append(x_train)
        y_trains.append(y_train)
        x_tests.append(x_test)
        y_tests.append(y_test)

    plot_vmd(u, data)
    return [x_trains, y_trains, x_tests, y_tests]


# 绘制分解后的序列图
def plot_vmd(u, data):
    recombine = np.sum(u, axis=0)
    # . Visualize decomposed modes
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), layout='constrained')
    ax1.plot(data, label='original')
    ax1.plot(recombine, label='reconstituted')
    ax1.set_title('Original signal and reconstituted signal')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Energy (kWh)')
    ax1.legend()
    ax1.grid()

    ax2.plot(u.T)
    ax2.set_title('Decomposed modes')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Energy (kWh)')
    ax2.legend(['Mode %d' % m_i for m_i in range(u.shape[0])])
    ax2.grid()
    plt.show()
    # plt.savefig("image/VMD.png")
    # plt.clf()
    # plt.cla()
    # plt.close(fig)


# normalization
def maponezero(data, direction="normal", maxmin=None):
    if direction == "normal":
        if maxmin != None:
            maxval = maxmin[0]
            minval = maxmin[1]
            data = (data - minval) / (maxval - minval)
        else:
            maxval = np.max(data)
            minval = np.min(data)
            data = (data - minval) / (maxval - minval)
        return [data, [maxval, minval]]
    if direction == "apply":
        maxval = maxmin[0]
        minval = maxmin[1]
        data = (data - minval) / (maxval - minval)
        return data
    if direction == "reverse":
        maxval = maxmin[0]
        minval = maxmin[1]
        data = data * (maxval - minval) + minval
        return data


def RMSE(predict_data, true_data):
    return np.sqrt(np.sum((true_data - predict_data) ** 2) / len(true_data))


def MAE(predict_data, true_data):
    return np.sum(np.abs(true_data - predict_data)) / len(true_data)


def MAPE(predict_data, true_data):
    return 100 * np.sum(np.abs((true_data - predict_data) / true_data)) / len(true_data)
