import numpy as np
from train_tf import train as fitness
import random
from MOEAD.normalization import normalize
from const import Parameters


def crossover_mutation(moead, p1, p2):
    y1 = np.copy(p1)
    y2 = np.copy(p2)

    # crossover
    if np.random.rand() < moead.c_rate:
        y1, y2 = crossover(moead, y1, y2)

    # # mutation
    # if np.random.rand()<moead.m_rate:
    #     y1=mutation(moead,y1)
    #     y2=mutation(moead,y2)
    y1 = mutation(moead, y1)
    y2 = mutation(moead, y2)

    # calculate the fitness of the new individual
    mse_y1, std_y1 = fitness(moead, y1)
    mse_y2, std_y2 = fitness(moead, y2)

    y1[moead.gene_num] = mse_y1
    y1[moead.gene_num+1] = std_y1

    y2[moead.gene_num] = mse_y2
    y2[moead.gene_num+1] = std_y2

    y1 = [y1]
    y2 = [y2]
    y1 = np.array(y1)
    y2 = np.array(y2)

    normalize(moead, y1)
    normalize(moead, y2)

    return y1[0], y2[0]


# two-point crossover
def crossover(moead, p1, p2):
    status = True
    # generate two crossover point
    while status:
        k1 = random.randint(0, moead.gene_num - 1)
        k2 = random.randint(0, moead.gene_num)
        if k1 < k2:
            status = False

    fragment1 = p1[k1: k2]
    fragment2 = p2[k1: k2]

    p1[k1: k2] = fragment2
    p2[k1: k2] = fragment1

    return p1, p2


# bitwise mutation
def mutation(moead, p):
    parameters = [
        Parameters.k,
        Parameters.seq_len,
        Parameters.batch_size,
        Parameters.learning_rate,
        Parameters.optimizer,
        Parameters.num_layer,
        Parameters.hidden_size,
        Parameters.dropout,
        Parameters.dense,
        Parameters.dense_unit]
    # mutation_pop = pop.copy()
    print('before mutation:', list(p))
    for j in range(moead.gene_num):
        if random.random() < moead.m_rate:
            new_gene = parameters[j][random.randint(0, len(parameters[j]) - 1)]
            # mutation_point = random.randint(0, gene_num - 1)
            p[j] = new_gene
    print('after mutation', list(p))
    return p
