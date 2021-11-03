#import os
#import tempfile
from matplotlib import pyplot as plt
import numpy as np
import pyabc


import seir_abm
import progressbar
from time import sleep


class ABM_param_estim:
    def __init__(self, n_agents, n_steps, cont_net, b0, trans_dists):
        """ Configuration """
        self.n_cores = 4

        """ ABM Environment variables"""
        self.n_states = 128
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.cont_net = cont_net
        self.b0 = b0
        self.trans_dists = trans_dists

        """ ABC-SMC variables """
        #pyabc.settings.set_figure_params('pyabc')  # for beautified plots
        self.db_path_abc_res = "sqlite:///" + "./data_of_runs.db"

    """ ABC-SMC routines """

    def seir_model_b0(self, parameter):
        mdl = seir_abm.ABM_env(n_agents=self.n_agents, n_steps=self.observations.size,
                               cont_net=self.cont_net, b0=self.b0,
                               trans_dists=self.trans_dists)
        return {"data": mdl.run_b0(parameter["mean"], self.observations.size)}

    def seir_model_all(self, parameters):
        mdl = seir_abm.ABM_env(n_agents=self.n_agents, n_steps=self.observations.size,
                               cont_net=self.cont_net, b0=self.b0,
                               trans_dists=self.trans_dists)
        return {"data": mdl.run_all_params(parameters, self.observations.size)}

    def distance_metric(self, x, y):
        dist = x["data"] - y["data"]
        dist = np.sqrt(np.sum(dist**2))
        return dist

    def find_b0_abc_smc(self, observations, lb, lu, min_ep=.1, pop_size=10, max_pop=10):
        self.lb, self.lu  = lb, lu
        self.observations = observations
        self.prior = pyabc.Distribution(mean=pyabc.RV("uniform", self.lb, self.lu - self.lb))
        abc = pyabc.ABCSMC(self.seir_model_b0, self.prior, self.distance_metric, population_size=pop_size)
        abc.new(self.db_path_abc_res, {"data": observations})
        abc.sampler.show_progress = True
        self.abc_history = abc.run(minimum_epsilon=min_ep, max_nr_populations=max_pop)

    def find_all_abc_smc(self, observations, lbs, lus, min_ep=.1, pop_size=10, max_pop=10):
        self.lbs, self.lus = lbs, lus
        self.lb_b0, self.lu_b0  = lbs[0], lus[0]
        self.lb_e2i, self.lu_e2i = lbs[1], lus[1]
        self.lb_i2r, self.lu_i2r = lbs[2], lus[2]

        self.observations = observations
        self.prior = pyabc.Distribution(b0_unif=pyabc.RV("uniform", self.lb_b0, self.lu_b0 - self.lb_b0),
                                        e2i_exp=pyabc.RV("uniform", self.lb_e2i, self.lu_e2i - self.lb_e2i),
                                        i2r_exp=pyabc.RV("uniform", self.lb_i2r, self.lu_i2r - self.lb_i2r))

        abc = pyabc.ABCSMC(self.seir_model_all, self.prior, self.distance_metric, population_size=pop_size)
        abc.new(self.db_path_abc_res, {"data": observations})
        self.abc_history = abc.run(minimum_epsilon=min_ep, max_nr_populations=max_pop)

    def plot_abc_res_b0(self):
        fig, ax = plt.subplots()
        res_means = np.zeros(self.abc_history.max_t + 1)
        for t in range(self.abc_history.max_t + 1):
            df, w = self.abc_history.get_distribution(m=0, t=t)
            res_means[t] = np.sum(df.to_numpy().squeeze(1) * w)
            pyabc.visualization.plot_kde_1d(
                df, w,
                xmin=self.lb, xmax=self.lu,
                x="mean", ax=ax,
                label="PDF t={}".format(t))
        ax.axvline(np.mean(res_means), color="k", linestyle="dashed")
        ax.legend()
        plt.show()

    def plot_abc_res_all(self):
        df, w = self.abc_history.get_distribution(m=0)
        pyabc.visualization.plot_kde_matrix(df, w)
        plt.show()

    def get_mean_all(self):
        means = np.zeros((3, ))
        stds = np.zeros((3, ))
        for p in range(3):
            res_means = np.zeros(self.abc_history.max_t + 1)
            for t in range(self.abc_history.max_t + 1):
                df, w = self.abc_history.get_distribution(m=0, t=t)
                res_means[t] = np.sum(df.to_numpy()[:, p] * w)
            means[p], stds[p] = np.mean(res_means), np.std(res_means)
        return means, stds

    def get_mean_b0(self):
        res_means = np.zeros(self.abc_history.max_t + 1)
        for t in range(self.abc_history.max_t + 1):
            df, w = self.abc_history.get_distribution(m=0, t=t)
            res_means[t] = np.sum(df.to_numpy().squeeze(1) * w)
        return np.mean(res_means), np.std(res_means)

    def get_dist_pdfs(self):
        fig, ax = plt.subplots()
        res_means = np.zeros(self.abc_history.max_t + 1)
        all_x_valls, all_pdfs = list(), list()
        for t in range(self.abc_history.max_t + 1):
            df, w = self.abc_history.get_distribution(m=0, t=t)
            res_means[t] = np.sum(df.to_numpy().squeeze(1) * w)
            x_vals, pdf = pyabc.visualization.kde.kde_1d( df, w,
                xmin=self.lb, xmax=self.lu,
                x="mean")
            all_x_valls.append(x_vals)
            all_pdfs.append(pdf)
        return all_x_valls, all_pdfs




