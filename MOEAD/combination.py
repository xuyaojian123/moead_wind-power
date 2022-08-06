from MOEAD.fast_nondominated_sort import fast_non_dominated_sort as Sort
from MOEAD.fast_nondominated_sort import crowding_distance as Distance
import numpy as np


# def combination(P, C, individual_num, gene_num):
def combination(P, C, moead):
    individual_num=moead.individual_num
    gene_num=moead.gene_num

    P = np.concatenate((P, C), axis=0)
    Pop_distance = Distance(P, len(P), gene_num)
    front,rank = Sort(Pop_distance, gene_num)


    # print(rank)

    P_temp = []

    for i in range(len(front)):
        if (len(P_temp) < individual_num):
            if (individual_num - len(P_temp)) >= len(front[i]):
                for j in range(len(front[i])):
                    P_temp.append(Pop_distance[front[i][j], :])
            else:
                # pop_distance = Distance(P, len(P), gene_num)
                pop_temp = []
                for j in front[i]:
                    pop_temp.append(Pop_distance[j])
                pop_temp = sorted(pop_temp, key=lambda x: x[gene_num + 2],reverse=True)
                pop_temp = np.array(pop_temp)
                # pop_temp = np.delete(pop_temp, 10, axis=1)

                if (individual_num - len(P_temp)) > (len(front[i])):
                    for index in range(len(front[i])):
                        P_temp.append(pop_temp[index, :])
                else:
                    for index in range(individual_num - len(P_temp)):
                        P_temp.append(pop_temp[index, :])
        else:
            break
    P_temp = np.array(P_temp)
    P_temp = np.delete(P_temp, gene_num+2, axis=1)
    # print(P_temp)

    return P_temp
