import pyabc

import numpy as np
from timeit import default_timer as timer
from scipy.sparse import csr_matrix
import scipy.stats as sts
from seir_param_estim import ABM_param_estim
from datetime import datetime



""" Transition distributions """
b0 = 3.3/12.0/4.0
e2i_dist = sts.gamma(a=2.0, scale=2.0)
i2r_dist = sts.gamma(a=3.0, scale=2.0)
trans_dists = {'e2i': e2i_dist, 'i2r': i2r_dist}

""" ABC Run """
# Daily infection cases (data for fitting)
observs = np.array([1,1,0,0,0,6,2,1,2,1,3,3,
                    2,0,1,2,1,0,0,0,0,0,1,2,20,53,100,
                    229,169,231,144,284,505,571,813])

n_steps = observs.size

# Hyper parameters
b0_lb = 0.01 #0.1/12.0/4.0    # lower search bound
b0_lu = 0.1 #7.0/12.0/4.0    # upper search bound
abc_pop_size = 1000       # Number of particles in each generation
abc_max_pop = 7           # Number of generations
abc_eps = 0.3             # Stopping crit
# String results
varx_all = list()
pdfs_all = list()
n_files = 1
file_arr  = np.array([10]) - 1
# file_arr  = np.array([1, 2, 5, 7, 10]) - 1

for file in file_arr: #range (n_files):
    """ Loading the contact data from the file """
    print("Loading the file # " + str(file + 1))
    file_path = "./scnet/scnet_" + str(file+1) + "0000.txt"
    data = np.genfromtxt(file_path)
    rows = data[:, 0]-1
    cols = data[:, 1]-1
    values = np.repeat(True, cols.shape[0])
    n_agents = (np.max(cols) + 1).astype(int)
    cont_matrix = csr_matrix((values, (rows, cols)), (n_agents, n_agents), dtype=bool)

    """ Run optimization """
    print("Running optimization for file # " + str(file + 1))
    abm_param_est = ABM_param_estim(n_agents=n_agents, n_steps=n_steps, cont_net=cont_matrix,
                                      b0=b0, trans_dists=trans_dists)
    start = timer()
    abm_param_est.find_b0_abc_smc(observations=observs, lb=b0_lb, lu=b0_lu,
                                   pop_size=abc_pop_size, max_pop=abc_max_pop, min_ep=abc_eps)
    end = timer()

    """ Retrieving results """
    var_x, pdfs = abm_param_est.get_dist_pdfs()
    varx_all = varx_all + var_x
    pdfs_all = pdfs_all + pdfs
    print("Execution time: {}".format(end - start))
    print()

""" Saving results"""
varx_mtrx = np.stack(varx_all)
pdfs_mtrx = np.stack(pdfs_all)

now = datetime.now()
date_time = now.strftime("%m:%d:%Y:%H:%M:%S")
np.savetxt('./results/varx_mtrx_' + date_time +'.txt', varx_mtrx, delimiter=' ')
np.savetxt('./results/pdfs_mtrx_' + date_time +'.txt', pdfs_mtrx, delimiter=' ')
