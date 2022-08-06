# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# @File  : train.py
# @Author: XuYaoJian
# @Date  : 2022/7/13 16:25
# @Desc  :
# """
# import numpy as np
# import torch
# from torch import nn
#
# from utils import load_data_with_vmd, maponezero
# from dataset import PowerDataset
# from torch.utils.data import DataLoader
# from model_build import create_model
# from test import evaluate
#
# # Define your execution device
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#
# def train_models(models, pop):
#     mses = []
#     for i, model in enumerate(models):
#         mse = train(model, i, pop[i])
#         mses.append(mse)
#     return np.array(mses)
#
#
# # 返回k个子序列的训练过程中，在验证集上表现最好的k个loss，然后取平均值
# def train(model, individual_index, individual):
#     print("The model will be running on", device, "device")
#     # 损失函数
#     criterion = nn.MSELoss()
#     EPOCHS = 200
#
#     # 学习率
#     learning_rate = int(individual[3])
#     if learning_rate == 0:
#         learning_rate = 0.005
#     elif learning_rate == 1:
#         learning_rate = 0.01
#     elif learning_rate == 2:
#         learning_rate = 0.05
#     elif learning_rate == 3:
#         learning_rate = 0.1
#     elif learning_rate == 4:
#         learning_rate = 0.5
#
#     # 优化器
#     optimizer = int(individual[4])
#     if optimizer == 0:
#         optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     elif optimizer == 1:
#         optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
#     elif optimizer == 2:
#         optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
#     # batch_size
#     batch_size = int(individual[2])
#     size = [32, 64, 128, 256]
#     batch_size = size[batch_size]
#     # 分解个数k
#     vmd_k = int(individual[0])
#     # 序列步长
#     seq_len = int(individual[1])
#     filename = '../data/data.csv'
#     [X_trains, Y_trains, X_tests, Y_tests] = load_data_with_vmd(filename, vmd_k, seq_len)
#
#     x_trains = []
#     y_trains = []
#     x_tests = []
#     y_tests = []
#     x_maxmins = []
#     y_maxmins = []
#     for k in range(vmd_k):
#         # normalize data
#         [x_train, x_maxmin] = maponezero(X_trains[k])
#         [y_train, y_maxmin] = maponezero(Y_trains[k])
#         x_test = maponezero(X_tests[k], "apply", x_maxmin)
#         y_test = maponezero(Y_tests[k], "apply", y_maxmin)
#         x_trains.append(x_train)
#         y_trains.append(y_train)
#         x_tests.append(x_test)
#         y_tests.append(y_test)
#         x_maxmins.append(x_maxmin)
#         y_maxmins.append(y_maxmin)
#
#     # 分解成k个序列，分别有k个模型训练
#     models = [model]
#     for i in range(1, vmd_k):
#         models.append(create_model(individual))
#     # Convert model parameters and buffers to CPU or Cuda
#     for mo in models:
#         mo.to(device)
#
#     best_val_loss = [np.inf for x in range(vmd_k)]
#     predict_test_sum = 0
#     for k in range(vmd_k):
#         train_data = PowerDataset(x_trains[k], y_trains[k])
#         train_size = int(len(train_data) * 0.75)
#         validate_size = len(train_data) - train_size
#         train_data, validation_data = torch.utils.data.random_split(train_data, [train_size, validate_size])
#         # test_data = PowerDataset(x_tests[k], y_tests[k])
#         train_iter = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
#         validation_iter = DataLoader(validation_data, batch_size=batch_size, shuffle=True, drop_last=False)
#         # test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False, drop_last=False)
#         for epoch in range(EPOCHS):
#             # Make sure gradient tracking is on, and do a pass over the data
#             models[k].train()
#             train_loss = 0.0
#             running_loss = 0.0
#             epoch_loss = 0.0
#             for i, data in enumerate(train_iter):
#                 # Every data instance is an input + label pair
#                 x = data[0].float().to(device)
#                 y = data[1].float().to(device)
#                 # Zero your gradients for every batch!
#                 optimizer.zero_grad()
#                 # Make predictions for this batch
#                 outputs = models[k](x)
#                 # Compute the loss and its gradients
#                 loss = criterion(outputs, y)
#                 loss.backward()
#                 # Adjust learning weights
#                 optimizer.step()
#                 # Gather data and report
#                 train_loss += loss.item()
#                 epoch_loss += loss.item()
#                 if i % 10 == 9:
#                     running_loss = train_loss / 10  # loss per batch
#                     # print(f'[vmd:{k + 1} , epoch: {epoch + 1}, batch: {i + 1}] loss: {running_loss:.3f}')
#                     train_loss = 0.0
#             # epoch_loss = epoch_loss / len(train_data)
#             epoch_loss = epoch_loss / (i + 1)
#
#             # We don't need gradients on to do reporting
#             models[k].eval()
#             running_vloss = 0.0
#             for i, data in enumerate(validation_iter):
#                 vx = data[0].float().to(device)
#                 vy = data[1].float().to(device)
#                 voutputs = models[k](vx)
#                 vloss = criterion(voutputs, vy)
#                 running_vloss += vloss.item()
#             # running_vloss = running_vloss / len(validation_data)
#             running_vloss = running_vloss / (i + 1)
#
#             if epoch % 1 == 0:
#                 print("|种群第{}个个体 | vmd: {} |, Epoch {}: | Train loss: {:.3f} | valid loss: {:.3f}".format(
#                     individual_index + 1, k, epoch + 1, epoch_loss, running_vloss))
#
#             # During training, save the model with the lowest loss on the validation set (this is correct)
#             if running_vloss < best_val_loss[k]:
#                 best_val_loss[k] = running_vloss
#
#         # 在测试集上验证效果
#         models[k].eval()
#         x_tests[k] = torch.from_numpy(x_tests[k]).float().to(device)
#         predict_sequence = models[k](x_tests[k])
#         predict_sequence_reverse = maponezero(predict_sequence, 'reverse', y_maxmins[k])
#         predict_test_sum += predict_sequence_reverse
#
#     return cal_gap(predict_test_sum, torch.from_numpy(Y_tests[vmd_k]).float().to(device))
#
#
# def cal_gap(y_hat, y):
#     fn = nn.MSELoss(reduction='sum')
#     gap = fn(y_hat, y).item()
#     return gap
