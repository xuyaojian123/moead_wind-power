# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# @File  : model_build.py
# @Author: XuYaoJian
# @Date  : 2022/7/5 10:37
# @Desc  : 根据参数定义网络模型 GRU + Linear
# """
# from torch import nn
# import torch
#
# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, GRU
# from tensorflow.keras import Input
# from tensorflow.keras.optimizers import Adam, SGD, Adamax
# from keras import backend as K
# from tensorflow.keras.losses import MeanSquaredError
#
# # Define your execution device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# # k                 pop[i][0]     2,3,4,5
# # seq_len           pop[i][1]     6,7,8,9,10,11,12,13
# # batch_size        pop[i][2]     32,64,128,256
# # learning_rate     pop[i][3]     0.005,0.01,0.05,0.1,0.5
# # optimize          pop[i][4]     Adam，SGD，Adamax
# # num_layer         pop[i][5]     5 1,2,3
# # num_layer_unit    pop[i][6]     30,31,32,33,34,35,36,37,38,39,40
# # dropout           pop[i][7]     0,0.25,0.5
# # dense             pop[i][8]     1,2,3
# # dense_unit        pop[i][9]     10,15,20,25,30
#
# # 创建一个种群的model
# # def create_pop_model(pop):
# #     pop_size = pop.shape[0]
# #     models = []
# #     for i in range(pop_size):
# #         individual = pop[i]
# #         models.append(GRULinear(individual))
# #     return models
# def create_pop_model(pop):
#     pop_size = pop.shape[0]
#     models = []
#     for i in range(pop_size):
#         individual = pop[i]
#         models.append(model_build(individual))
#     return models
#
#
# # 创建一个model
# # def create_model(individual):
# #     model = GRULinear(individual)
# #     return model
# def create_model(individual):
#     model = model_build(individual)
#     return model
#
#
# def model_build(individual):
#     # 建立模型
#     seq_len = int(individual[1])
#     learning_rate = individual[3]
#     # 优化器
#     optimizer = int(individual[4])
#     if optimizer == 0:
#         optimizer = Adam(learning_rate=learning_rate)
#     elif optimizer == 1:
#         optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
#     elif optimizer == 2:
#         optimizer = Adamax(learning_rate=learning_rate)
#     num_layers = int(individual[5])
#     hidden_size = int(individual[6])
#     dropout = individual[7]
#     dense = int(individual[8])
#     dense_units = int(individual[9])
#     model = Sequential()
#     model.add(Input(shape=(seq_len, 1)))
#     for i in range(num_layers - 1):
#         model.add(GRU(units=hidden_size, return_sequences=True, kernel_initializer='glorot_normal', dropout=dropout))
#     model.add(GRU(units=hidden_size, return_sequences=False, kernel_initializer='glorot_normal', dropout=dropout))
#     units = dense_units
#     for i in range(dense):
#         model.add(Dense(units, activation='relu', kernel_initializer='he_normal'))
#         units = int(units / 2)
#     model.add(Dense(1, activation='relu', kernel_initializer='he_normal'))
#     loss_fn = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
#     # model.compile(optimizer=optimizer, loss='mse')
#     model.compile(optimizer=optimizer, loss=loss_fn)
#     return model
#
# # 模型搭建(使用torch创建的模型训练一直没用)
# # class GRULinear(nn.Module):
# #     def __init__(self, individual):
# #         super().__init__()
# #         num_layers = int(individual[5])
# #         hidden_size = int(individual[6])
# #         dropout = int(individual[7])
# #         if dropout == 0:
# #             dropout = 0.
# #         elif dropout == 1:
# #             dropout = 0.25
# #         elif dropout == 2:
# #             dropout = 0.5
# #         # 当GRU只有一层时,设置dropout不起作用
# #         if num_layers == 1:
# #             dropout = 0
# #         linear_num = int(individual[8])
# #         linear_units_num = int(individual[9])
# #         dense_unit = [10, 15, 20, 25, 30]
# #         linear_units_num = dense_unit[linear_units_num]
# #
# #         self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
# #                           batch_first=True)
# #         self.linear1 = nn.Linear(in_features=hidden_size, out_features=linear_units_num)
# #         # 要用GPU训练的话，这里必须加上device。在model.to(device)时，这里创建的线性层却没有在GPU上，只能手动添加
# #         self.fcs = [nn.Linear(linear_units_num, linear_units_num, device=device) for _ in range(linear_num)]
# #         self.fi = nn.Linear(in_features=linear_units_num, out_features=1)
# #
# #     def forward(self, x):
# #         """
# #         :param x:[batch,seq_len,input_size] => [batch_size,seq_len,1]
# #         :return:
# #         """
# #         out, hn = self.gru(x)
# #         out = out[:, -1, :]  # 只是用最后一个时间步的输出
# #         out = self.linear1(out)
# #         out = torch.sigmoid(out)
# #         for fc in self.fcs:
# #             out = fc(out)
# #             out = torch.sigmoid(out)
# #         out = self.fi(out)
# #         return out
