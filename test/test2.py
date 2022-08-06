import numpy as np
import pandas as pd
from vmdpy import VMD

# data = pd.read_csv('data/week_data/data/washingtong_winter_20120101-20120107新.csv')['wind speed at 100m (m/s)']
# data = np.array(data).reshape(-1)
# # some parameters for VMD
# alpha = 2000  # moderate bandwidth constraint
# DC = 0  # no DC part imposed
# init = 1  # initialize omegas uniformly
# tol = 1e-7  # tolerance 0.5
# REI = np.inf
# tauo = 0
# # 寻找最佳的tau值
# tau = tauo
# u, u_hat, omega = VMD(data, alpha, tau, 3, DC, init, tol)
# a  = 1

data = [1,2,3,4]
y = [2,4,5,6]
data = np.array(data)
y = np.array(y)
df = pd.DataFrame(data={
    'data': data,
    'y': y
})
df.to_csv('result/a/1.csv')
