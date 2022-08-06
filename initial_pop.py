from MOEAD.pop_init import pop_init
from model_build import model_build
from train_tf import train as fitness
import pandas as pd
import numpy as np
from MOEAD.fast_nondominated_sort import fast_non_dominated_sort as Sort


def initial_pop(moead):
    print(
        '***************************************************************************************************************************')
    print(
        '**************************************************** Initialization *******************************************************')

    # ############ 1.generate initial population #########################
    # randomly generate population
    pop = pop_init(moead.individual_num, moead.gene_num)
    # build model and calculate the fitness
    for i in range(moead.individual_num + 1):
        print("Initialization: The %dth individual" % (i + 1))
        mse, std = fitness(moead, pop[i])
        pop[i][moead.gene_num] = mse
        pop[i][moead.gene_num + 1] = std
    # save the initial population
    # print(pop)
    inv_y = pd.DataFrame(pop)
    inv_y.to_csv(moead.save_filename+'pop_init_1.csv', header=False, index=False)

    ################ 2.load existed initial population######################
    # pop = pd.read_csv('result/pop_init_1.csv', header=None)
    # pop = np.array(pop)
    return pop
