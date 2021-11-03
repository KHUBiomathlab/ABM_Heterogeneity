import multiprocessing

from matplotlib import pyplot as plt
import numpy as np
from numpy.ma.bench import timer

import seir_abm


class ABM_batch_run:
    def __init__(self, n_agents, n_steps, cont_net, b0, trans_dists):
        """ Configuration """
        self.n_cores = 4
        self.batch_size = 0

        """ ABM Environment variables"""
        self.n_states = 4
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.cont_net = cont_net
        self.b0 = b0
        self.trans_dists = trans_dists

    """ Multi-batch running of the ABM"""
    def run_sim(self, run_id):
        mdl = seir_abm.ABM_env(n_agents=self.n_agents, n_steps=self.n_steps,
                               cont_net=self.cont_net, b0=self.b0,
                               trans_dists=self.trans_dists, run_id=run_id)
        return mdl.run()

    def run_batch(self, batch_size):
        self.batch_size = batch_size
        pool = multiprocessing.Pool(processes=self.n_cores, initializer=np.random.seed)
        result_async = [pool.apply_async(self.run_sim, args=(i,)) for i in
                        range(batch_size)]
        self.new_inf_curves = [r.get() for r in result_async]
        return self.new_inf_curves

    """ Plots """
    def plot_all_curves(self):
        for curve in self.new_inf_curves:
            plt.plot(curve, 'b', linewidth=0.8)
        res_mat = np.stack(self.new_inf_curves)
        res_mean = np.mean(res_mat, axis=0)
        res_median = np.median(res_mat, axis=0)
        plt.plot(res_mean, 'r', linewidth=1.2)
        plt.plot(res_median, 'g', linewidth=1.2)
        #plt.legend(["Run", "Mean", "Median"])
        plt.show()


