#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : MyProblem.py
@Author: XuYaoJian
@Date  : 2022/7/5 10:37
@Desc  : 
"""
import geatpy as ea
import numpy as np
from model_build import create_pop_model
from utils import get_parameter_number,count_params
# from train import train_models
from train_tf import train


class MyProblem(ea.Problem):
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        Dim = 10  # 初始化Dim（决策变量维数）
        maxormins = [-1, -1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [1] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [2, 6, 0, 0, 0, 1, 30, 0, 1, 0]  # 决策变量下界
        ub = [5, 13, 3, 6, 2, 3, 40, 2, 4, 4]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化，两个目标需要优化
        ea.Problem.__init__(self, name, 2, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    # # 重写评价函数,f1目标为模型复杂度，即为模型的参数数量。f2为预测指标的精度
    # def evalVars(self, Vars):  # 目标函数,Vars为二维矩阵，行表示种群大小，列表示每个个体变量取值
    #     pop_size = Vars.shape[0]
    #     for i in range(0,pop_size):
    #         train(Vars[i])
    #
    #
    #     models = create_pop_model(Vars)
    #     f1 = []
    #     for model in models:
    #         f1.append(count_params(model))
    #     f1 = np.array(f1).reshape(-1, 1)
    #     f2 = train_models(models, Vars).reshape(-1, 1)
    #     ObjV = np.hstack([f1, f2])  # 计算目标函数值矩阵
    #     return ObjV

    # 重写评价函数,f1目标为模型复杂度，即为模型的参数数量。f2为预测指标的精度
    def evalVars(self, Vars):  # 目标函数,Vars为二维矩阵，行表示种群大小，列表示每个个体变量取值
        pop_size = Vars.shape[0]
        f1 = []
        f2 = []
        for i in range(0, pop_size):
            mse, params = train(Vars[i])
            f1.append(mse)
            f2.append(params)
        f1 = np.array(f1).reshape(-1, 1)
        f2 = np.array(f2).reshape(-1, 1)
        ObjV = np.hstack([f1, f2])  # 计算目标函数值矩阵
        return ObjV
