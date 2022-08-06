from MOEAD.MOEAD_Utils import *
import random
from MOEAD.normalization import normalize


def update_pop(P, moead, P_B, Y):
    # update Max and Min
    # a = P.max(axis=0)
    # b = P.min(axis=0)
    # Max, Min = a[9], b[9]
    # if Y[9] > Max:
    #     Max = Y[9]
    # if Y[9] < Min:
    #     Min = Y[9]
    # moead.Max = Max
    # moead.Min = Min

    # normalize(moead, P)

    # 根据Y更新P_B集内邻居
    for j in P_B:
        Xj = P[j]
        d_x = cpt_tchbycheff(moead, j, Xj)
        d_y = cpt_tchbycheff(moead, j, Y)
        if d_y <= d_x:
            # d_y 的切比雪夫距离更小
            # P[j] = Y[:]
            if P[j][moead.gene_num] > Y[moead.gene_num]:
                P[j] = Y[:]
            elif random.random() > 0.7:
                P[j] = Y[:]
            # if P[j][8] > Y[8]:
            #     P[j] = Y[:]
            # elif random.random() > 0.80:
            #     P[j] = Y[:]

            # F_Y = moead.Test_fun.Func(Y)
            # moead.Pop_FV[j] = F_Y
            # update_EP_By_ID(moead, j, F_Y)

        # pass
# def update_EP_By_ID(moead, id, F_Y):
#     # 如果id存在，则更新其对应函数集合的值
#     if id in moead.EP_X_ID:
#         # 拿到所在位置
#         position_pi = moead.EP_X_ID.index(id)
#         # 更新函数值
#         moead.EP_X_FV[position_pi][:] = F_Y[:]
