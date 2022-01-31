import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy import stats
from util import *

class UCBPI_pull():
    def __init__(self, price_list, segments, c, bias, UCB1):
        '''
        c is a hyperparameter constant, and is default to be 2
        '''
        self.t = 0
        self.K = len(price_list)
        self.price_list = price_list
        self.criteria = np.zeros(self.K)
        self.mean_profit = np.zeros(self.K)
        self.c = c
        self.Cum_reward = np.zeros(self.K)
        self.Pull_times = np.zeros(self.K)
        self.reward = np.array([])
        self.arm_sequence = np.array([])
        self.seg_num = segments
        self.UCB1 = UCB1
        # seg_size can be adjusted to be unequal sizes
        self.seg_size = np.repeat(1/self.seg_num,self.seg_num)
        # Demand learning
        self.nu_s_t = np.repeat((min(self.price_list)+max(self.price_list))/2, self.seg_num)
        self.delta = np.repeat((min(self.price_list)+max(self.price_list))/2, self.seg_num)
        self.p_min = np.repeat(min(self.price_list), self.seg_num) 
        self.p_max = np.repeat(max(self.price_list), self.seg_num)
        self.bias = bias
        self.delta_hat = max(self.delta)
    
    def UCBPIAction(self, PI=True):
        '''
        Select action according to UCB Criteria
        The main strategy
        UCB original or amended
        '''
        if self.UCB1:
            self.criteria = (self.mean_profit + np.sqrt( self.c * (np.log(self.t+1)) / (self.Pull_times+1)))
            
        else:
            self.criteria = (self.mean_profit + self.price_list * np.sqrt( self.c * (np.log(self.t+1)) / (self.Pull_times+1)))
        
        # Partial Intification
        if PI == True:
            LB = np.zeros(self.K)
            UB = np.zeros(self.K)
            for arm, price in enumerate(self.price_list):
                LB[arm] = price * np.dot(self.seg_size,(self.p_min >= price)).sum()
                UB[arm] = price * np.dot(self.seg_size,(self.p_max >= price)).sum()
            self.criteria[UB <= max(LB)] = 0
        else:
            pass
        
        arm = rd_argmax(self.criteria)
        return arm
    
    def update_lists(self, arm, pricing_MAB):
        '''
        Update lists and draw sample regarding to the experiment environments in each round
        '''
        new_reward, chosen_segments_index, reaction = pricing_MAB.customer_simulation(self.price_list[arm])
        self.Pull_times[arm] = self.Pull_times[arm] + 1
        self.arm_sequence = np.append(self.arm_sequence, arm)
        self.reward = np.append(self.reward, new_reward)
        self.Cum_reward[arm] = self.Cum_reward[arm] + new_reward
        self.mean_profit[arm] = self.mean_profit[arm] + ( new_reward - self.mean_profit[arm] ) / self.Pull_times[arm]
        
        #Demand learning
        # Update price bounds for segments
        # p_min is the known highest price that customer from segments s won't purchase
        # p_max is the known lowest price that customer from segments s will all purchase
        self.p_max[chosen_segments_index[~reaction]] = np.maximum(self.p_max[chosen_segments_index[~reaction]],
                                                      self.price_list[arm])
        self.p_min[chosen_segments_index[reaction]] = np.minimum(self.p_min[chosen_segments_index[reaction]],
                                                      self.price_list[arm])
        # Update nu_s_t(estimated ideal pricing for segments s)
        self.nu_s_t = (self.p_min+self.p_max)/2
        # Update delta(estimated segments within variance)
        self.delta = (self.p_max-self.p_min)/2
        # Update delta_hat
        self.delta_hat = max(self.delta) + self.bias
        self.t += 1
        
    def UCBPI(self, T, pricing_MAB):
        '''
        Implementation of the whole experiment
        T: number of time horizon
        pricing_MAB: the fixed experiment environment
        '''
        while self.t < T:
            arm = self.UCBPIAction()
            self.update_lists(arm, pricing_MAB)
        return self.Cum_reward, self.Pull_times, self.reward, self.arm_sequence


class TS_pull():
    
    def __init__(self, price_list, est='MEAN'):
        '''
        price_list: the action set, which refers to the list of price options
        est: how to do post estimation in bayesian setting, if default mean: post_sample ~ binom(N=10, p=E(theta))
        '''
        self.t = 0
        self.price_list = price_list
        self.theta = np.linspace(0, 1, 1001)
        self.K = len(price_list)
        self.p = np.tile(np.repeat(1/len(self.theta), len(self.theta)), (self.K,1))
        self.Cum_reward = np.zeros(self.K)
        self.Pull_times = np.zeros(self.K)
        self.reward = np.array([])
        self.arm_sequence = np.array([])
        self.est = est
        
    def TSAction(self):
        
        
        #MEAN Estimation
        if self.est == 'MEAN':
            post_theta = np.zeros(self.K)
            post_sample = np.zeros(self.K)
            post_reward = np.zeros(self.K)
            for i in range(self.K):
                post_theta[i] = np.dot(self.theta, self.p[i])
                post_sample[i] = np.random.binomial(10, post_theta[i], 1)
                post_reward[i] = post_sample[i] * self.price_list[i]
                
        #MAP Estimation
        if self.est == 'MAP':
            post_theta = np.zeros(self.K)
            post_sample = np.zeros(self.K)
            post_reward = np.zeros(self.K)
            for i in range(self.K):
                post_theta[i] = self.theta[rd_argmax(self.p[i])]
                post_sample[i] = np.random.binomial(10, post_theta[i], 1)
                post_reward[i] = post_sample[i] * self.price_list[i]
                
        else:
            post_theta = np.zeros(self.K)
            post_sample = np.zeros(self.K)
            post_reward = np.zeros(self.K)
            for i in range(self.K):
                post_theta[i] = np.random.choice(a=self.theta, size=1, p=self.p[i])
                post_sample[i] = np.random.binomial(10, post_theta[i], 1)
                post_reward[i] = post_sample[i] * self.price_list[i]
                
        
        arm = rd_argmax(post_reward)
        
        return arm
        
    def update_lists(self, arm, pricing_MAB):
        
        new_reward, chosen_segments_index, reaction = pricing_MAB.customer_simulation(self.price_list[arm])
        self.Pull_times[arm] = self.Pull_times[arm] + 1
        self.arm_sequence = np.append(self.arm_sequence, arm)
        self.reward = np.append(self.reward, new_reward)
        self.Cum_reward[arm] = self.Cum_reward[arm] + new_reward
        self.t += 1
        
        return reaction
        
    def update_prior(self, arm, reaction):
        
        observation = sum(reaction)
        likelihood = stats.binom(n=10, p=self.theta).pmf(observation)
        self.p[arm] = self.p[arm] * likelihood / sum(self.p[arm] * likelihood)
        
        
    def TS(self, T, pricing_MAB):
        
        while self.t < T:
            arm = self.TSAction()
            reaction = self.update_lists(arm, pricing_MAB)
            self.update_prior(arm, reaction)
            
        return self.Cum_reward, self.Pull_times, self.reward, self.arm_sequence

class EG_pull():
    
    def __init__(self, price_list):
        '''
        price_list: the action set, which refers to the list of price options
        est: how to do post estimation in bayesian setting, if default mean: post_sample ~ binom(N=10, p=E(theta))
        '''
        self.t = 0
        self.e = 0.2
        self.K = len(price_list)
        self.price_list = price_list
        self.Cum_reward = np.zeros(self.K)
        self.Pull_times = np.zeros(self.K)
        self.reward = np.array([])
        self.arm_sequence = np.array([])
        self.mean_profit = np.zeros(self.K)
        
    def EGAction(self):
        
        explore = np.random.choice([True, False],size=1, p=[self.e, 1-self.e])[0]
        
        if explore:
            arm = np.random.choice(np.arange(self.K))
        else:
            arm = rd_argmax(self.mean_profit)
        
        return arm
        
    def update_lists(self, arm, pricing_MAB):
        
        new_reward, chosen_segments_index, reaction = pricing_MAB.customer_simulation(self.price_list[arm])
        self.Pull_times[arm] = self.Pull_times[arm] + 1
        self.arm_sequence = np.append(self.arm_sequence, arm)
        self.reward = np.append(self.reward, new_reward)
        self.Cum_reward[arm] = self.Cum_reward[arm] + new_reward
        self.mean_profit[arm] = self.mean_profit[arm] + ( new_reward - self.mean_profit[arm] ) / self.Pull_times[arm] 
        self.t += 1
        
        return reaction
        
        
    def EG(self, T, pricing_MAB, e):
        
        self.e = e
        
        while self.t < T:
            arm = self.EGAction()
            reaction = self.update_lists(arm, pricing_MAB)
            
        return self.Cum_reward, self.Pull_times, self.reward, self.arm_sequence