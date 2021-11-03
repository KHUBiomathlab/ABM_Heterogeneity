import pyabc

import numpy as np
import scipy.stats as st
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt

import  seir_abm
from scipy.sparse import csr_matrix
import scipy.stats as sts
from timeit import default_timer as timer
#%matplotlib inline
from seir_batch_run import ABM_batch_run
from seir_param_estim import ABM_param_estim

"""Loading the contact data from the file"""
data = np.genfromtxt("triplets_scnet28.txt")
rows = data[:, 0]-1
cols = data[:, 1]-1
values = np.repeat(True, cols.shape[0])
n_agents = (np.max(cols) + 1).astype(int)
cont_matrix = csr_matrix((values, (rows, cols)), (n_agents, n_agents), dtype=bool)

""" Transition distributions"""
b0 = 3.3/12.0/4.0
e2i_dist = sts.lognorm(scale=np.exp(1.63), s=0.5)
i2r_dist = sts.expon(loc=0, scale=4.0)
trans_dists = {'e2i': e2i_dist, 'i2r': i2r_dist}
num_steps = 60

""" Single Run"""
mdl = seir_abm.ABM_env(n_agents=n_agents, n_steps=num_steps,
                       cont_net=cont_matrix, b0=b0, trans_dists=trans_dists)
start = timer()
mdl.run()
end = timer()

""" Results"""
print("Execution time: {}".format(end - start))
mdl.plot_inf_curve()
