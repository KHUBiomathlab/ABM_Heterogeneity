
import numpy as np
import numpy.random as np_rnd
import matplotlib.pyplot as plt
from datetime import datetime
import random
import scipy.stats as sts

class ABM_env:
    """n_states - Number of states of the model (e.g., SEIR -> 4)
       cont_net - Contact network (sparse matrix) (e.g., scalefree)
       trans_dist - Array of trasition destributions with size (n_state - 1)
    """
    def __init__(self, n_agents, n_steps, cont_net, b0, trans_dists, run_id=0):
        self.n_states = 4
        self.n_agents = n_agents
        self.n_steps = n_steps
        self.cont_net = cont_net
        self.b0 = b0
        self.run_id = run_id

        """ data containers (progress and results)"""
        self.state_count = np.zeros((self.n_states, self.n_steps), dtype=int)
        self.agent_states = np.zeros((self.n_agents,), dtype=int)
        self.trans_delay = np.repeat(-1, (self.n_agents,))
        self.new_inf = np.zeros((self.n_steps,), dtype=int)
        self.new_rem = np.zeros((self.n_steps,), dtype=int)
        self.step_id = 0

        """ setting up initial condition"""
        self.set_init_cond_degree(tg_degree=12)

        """ Distributions """
        self.e2i_dist = trans_dists['e2i']
        self.i2r_dist = trans_dists['i2r']

    """ Setting initial condition (infected agent selected by its degree)"""
    def set_init_cond_degree(self, tg_degree):
        all_degrees = self.cont_net.sum(axis=1)
        cond_agents = np.where(all_degrees == tg_degree)

        # Number of agents having "tg_degree"
        num_agents = cond_agents[0].shape[0]
        agnt_idx = cond_agents[0][np.random.choice(num_agents, 1)]
        self.agent_states[agnt_idx] = 2

    def run_all_params(self, params, new_n_steps):
        random.seed(datetime.now())
        self.e2i_dist = sts.expon(loc=0, scale=params["e2i_exp"])
        self.i2r_dist = sts.expon(loc=0, scale=params["i2r_exp"])
        self.b0 = params["b0_unif"]
        self.n_steps = new_n_steps
        for stp in range(self.n_steps):
            self.step()
        return self.new_rem

    def run_b0(self, new_b0, new_n_steps):
        random.seed(datetime.now())
        self.b0 = new_b0
        self.n_steps = new_n_steps
        for stp in range(self.n_steps):
            self.step()
        return self.new_rem

    def run(self):
        random.seed(datetime.now())
        for stp in range(self.n_steps):
            self.step()
        return self.new_rem

    def plot_inf_curve(self):
        plt.plot(self.new_rem)
        plt.ylabel('Daily Infected')
        plt.show()

    def step(self):
        """ S to E transition """
        inf_agents = np.where(self.agent_states == 2)
        # Suseptable agents connected with infected
        susept_cand = np.array([])
        for inf_idx in inf_agents[0]:
            # contacts of the infected agent
            adj_conts = self.cont_net.getrow(inf_idx).indices
            susept_cand = np.append(susept_cand, adj_conts)

        # getting candidates to be exposed and number of
        # of their contacts with infected agents
        sus_cands_unq, n_conts = np.unique(susept_cand, return_counts=True)
        exp_probs = 1 - np.exp(-self.b0*n_conts)

        # sample exposed agents
        exp_agents=sus_cands_unq[np.logical_and(
            self.agent_states[sus_cands_unq.astype(int)] == 0,exp_probs >
                               np_rnd.rand(exp_probs.shape[0]))].astype(int)
        self.trans_delay[exp_agents]=0 # Next day switch state

        """ Do transitions (after delays) """
        # S to E
        agents_to_E = np.logical_and(self.trans_delay == 0, self.agent_states == 0)
        self.agent_states[agents_to_E] = 1
        self.trans_delay[agents_to_E] = np.ceil(self.e2i_dist.ppf(
            np_rnd.rand(np.count_nonzero(agents_to_E))))

        # E to I
        agents_to_I = np.logical_and(self.trans_delay == 0, self.agent_states == 1)
        self.agent_states[agents_to_I] = 2
        self.trans_delay[agents_to_I] = np.ceil(self.i2r_dist.ppf(
            np_rnd.rand(np.count_nonzero(agents_to_I))))
        self.new_inf[self.step_id] = np.count_nonzero(agents_to_I)

        # I to R
        agents_to_R = np.logical_and(self.trans_delay == 0, self.agent_states == 2)
        self.agent_states[agents_to_R] = 3
        self.new_rem[self.step_id] = np.count_nonzero(agents_to_R)

        # Others
        self.step_id = self.step_id + 1
        self.trans_delay = self.trans_delay - 1

