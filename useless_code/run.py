#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : run.py
@Author: XuYaoJian
@Date  : 2022/7/15 21:12
@Desc  : 程序运行入口
"""

import geatpy as ea

from useless_code.MyProblem import MyProblem

if __name__ == "__main__":
    # 实例化问题对象
    problem = MyProblem()
    # 构建算法
    algorithm = ea.moea_MOEAD_archive_templet(problem,
                                              ea.Population(Encoding='RI', NIND=20),
                                              MAXGEN=10,
                                              logTras=1)

    algorithm.neighborSize = 3
    algorithm.MAXSIZE = 30

    # 求解
    res = ea.optimize(algorithm, seed=5, verbose=True, drawing=1,
                      outputMsg=True, drawLog=False, saveFlag=True, dirName='./moea_MOEAD_archive_templet/moea_MOEAD_archive_templet result')
    print(res)
    # ea.moea_MOEAD_DE_templet
    # ea.moea_MOEAD_templet
    # ea.moea_MOEAD_archive_templet
