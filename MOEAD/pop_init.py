import numpy as np
import random
import pandas as pd

from const import Parameters


# generate initial population
def pop_init(pop_num, gene_num):
    pop = np.zeros([pop_num + 1, gene_num + 4])
    for i in range(0, pop_num + 1):
        k = Parameters.k[random.randint(0, len(Parameters.k) - 1)]
        seq_len = Parameters.seq_len[random.randint(0, len(Parameters.seq_len) - 1)]
        batch_size = Parameters.batch_size[random.randint(0, len(Parameters.batch_size) - 1)]
        learning_rate = Parameters.learning_rate[random.randint(0, len(Parameters.learning_rate) - 1)]
        optimizer = Parameters.optimizer[random.randint(0, len(Parameters.optimizer) - 1)]
        num_layer = Parameters.num_layer[random.randint(0, len(Parameters.num_layer) - 1)]
        hidden_size = Parameters.hidden_size[random.randint(0, len(Parameters.hidden_size) - 1)]
        dropout = Parameters.dropout[random.randint(0, len(Parameters.dropout) - 1)]
        dense = Parameters.dense[random.randint(0, len(Parameters.dense) - 1)]
        dense_unit = Parameters.dense_unit[random.randint(0, len(Parameters.dense_unit) - 1)]

        pop[i][0] = k
        pop[i][1] = seq_len
        pop[i][2] = batch_size
        pop[i][3] = learning_rate
        pop[i][4] = optimizer
        pop[i][5] = num_layer
        pop[i][6] = hidden_size
        pop[i][7] = dropout
        pop[i][8] = dense
        pop[i][9] = dense_unit

        pop[i][10] = 0.
        pop[i][11] = 0.
        pop[i][12] = 0.
        pop[i][13] = 0.
    # print(pop)
    # inv_y = pd.DataFrame(pop)
    # inv_y.to_csv('result/washingtong_winter_20120101-20120107/pop_init_1.csv', header=False, index=False)
    return pop
# pop_init(20,8)
