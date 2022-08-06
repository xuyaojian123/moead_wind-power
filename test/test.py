#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : test.py
@Author: XuYaoJian
@Date  : 2022/7/25 10:21
@Desc  : 
"""

import numpy as np

g = np.array([[1,2],[4,6]])

t = np.array([1,2])

k = (g - t)**2
print(k)

p = np.sum(k,axis=1)
print(p)

tt = [[1,2,3],[2,2,2]]
print(len(tt))
if __name__ == "__main__":
    pass
