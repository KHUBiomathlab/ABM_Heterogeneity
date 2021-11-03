import numpy as np
from scipy.sparse import csr_matrix
import scipy.stats as sts
from timeit import default_timer as timer
from seir_batch_run import ABM_batch_run

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

""" Batch Run"""
abm_batch = ABM_batch_run(n_agents=n_agents, n_steps=60,
        cont_net=cont_matrix, b0=b0, trans_dists=trans_dists)

start = timer()
results = abm_batch.run_batch(4)
end = timer()

""" Results """
print("Execution time: {}".format(end - start))
abm_batch.plot_all_curves()