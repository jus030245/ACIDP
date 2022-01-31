import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import seaborn as sns
from scipy import stats
from util import *


class IDS_pull():
    def __init__(self, price_list, N, start_L=2, update_L=2, simulate_time_initiate=1, simulate_time_update=1, window_width=300):
        """
        :param price_list: np.array, an array of prices aka actions
        :param N: number of possible outcomes
        :param start_L: int, number of thetas that one want to explore for
        :param update_L: number of thetas that one want to re-explore for, when environment is changed
        :param window_width: the width of window that one considered to be available recent data
        """
        #Given Context
        self.price_list = price_list
        self.K = len(price_list)
        self.N = N

        #Hyperparameters
        self.L = start_L
        self.update_L = update_L
        self.simulate_time_initiate = simulate_time_initiate
        self.simulate_time_update = simulate_time_update
        self.window_width = window_width
        self.alpha_cs = 0.05
        self.alpha_bb = 0.01
        self.cool_time = 20
        self.update_during_exploit = False
        
        #Initiate other parameters
        self.set_attribute()
    
    def set_attribute(self):
        #Initiated Parameters
        self.t = 0
        self.collection_rounds = 0
        self.IDS_rounds = 0
        self.Cum_reward = np.zeros(self.K)
        self.Pull_times = np.zeros(self.K)
        self.reward = np.array([])
        self.observation = np.array([])
        self.arm_sequence = np.array([])
        self.all_posterior = []
        self.obs = np.linspace(0, 1, self.N)
        self.window_on = False
        self.sign_count = 0
        self.sign = False
        self.detected_likelihood = np.array([])
        # self.lock_arm = False
        self.detector = {'trend':[], 'upper_bound':[], 'lower_bound':[]}
        self.detector_cool = float('inf')
        self.detected_stamp = None
        self.eg_on = False

        #Other
        self.store_IDS = True
        self.IDS_results = {'delta': [], 'g': [], 'IR': []}
        self.price_list_simu = ((self.price_list + np.roll(self.price_list, shift=-1)) / 2)[:-1]
        self.price_interval = np.mean(np.diff(self.price_list))
        
    def initiate_IDS_likelihood(self, pricing_MAB, simulate_time=1):
        #probability of thetas
        p = np.random.uniform(0, 1, self.L)
        p = p / p.sum()
        
        p_y = np.zeros(shape=(self.L, self.K, self.N))
        for theta in range(self.L):
            for arm, price in enumerate(self.price_list):
                for simulate_round in range(simulate_time):
                    # t = theta * self.K * simulate_time + arm * simulate_time + simulate_round
                    reaction = self.update_lists(arm, pricing_MAB)
                    self.all_posterior.append('initiate')
                    if self.store_IDS:
                        self.IDS_results['delta'].append('initiate')
                        self.IDS_results['g'].append('initiate')
                        self.IDS_results['IR'].append('initiate')
                    p_y[theta, arm, np.sum(reaction)] += 1 / simulate_time
                    
        #in order to prevent assuming lot of p_y element is 0, 1e-4 is plus for p_y
        p_y = p_y + 1e-4
        for i in range(p_y.shape[0]):
            for j in range(p_y.shape[1]):
                p_y[i,j] = p_y[i,j] / p_y[i,j].sum()

        #reward associated with each observation
        reward = self.price_list.reshape(self.K,1) @ np.arange(self.N).reshape(1,self.N)
        
        self.p, self.p_y, self.R = p, p_y, reward
            
    def update_likelihood(self, simulate_time=1):
        '''
        plus one universe each time triggered
        append larger p(theta) in self.p as new exploration are considered more useful
        the default prior for new likelihood is 0.8
        '''
        self.window_on = False
        self.L += 1
        #append initial p(theta) = 0.8 for new updated likelihood
        #such that new likelihood will be weighted more
        self.p = np.append(self.p,4)
        self.p = self.p / self.p.sum()
        
        new_p_y = np.zeros(shape=(1,self.K,self.N))
        for arm, price in enumerate(self.price_list):
            for simulate_round in range(simulate_time):
                reaction = self.update_lists(arm, self.pricing_MAB)
                self.all_posterior.append('update')
                if self.store_IDS:
                    self.IDS_results['delta'].append('update')
                    self.IDS_results['g'].append('update')
                    self.IDS_results['IR'].append('update')
                
                new_p_y[0, arm, np.sum(reaction)] += 1 / simulate_time
                self.collection_rounds += 1
        
        #in order to prevent assuming lot of p_y element is 0, 1e-4 is plus for p_y
        new_p_y = new_p_y + 1e-4
        for i in range(new_p_y.shape[0]):
            for j in range(new_p_y.shape[1]):
                new_p_y[i,j] = new_p_y[i,j] / new_p_y[i,j].sum()
                
        self.p_y = np.append(self.p_y,new_p_y,axis=0)
        self.window_on = True
        
    def initiate_window_likelihood(self):
        cloned_theta = rd_argmax(self.p)
        new_p_y = np.array([self.p_y[cloned_theta]])
        arm_window = self.arm_sequence[self.t-self.window_width:self.t]
        observation_window = self.observation[self.t-self.window_width:self.t]

        unique_arms, arm_counts = np.unique(arm_window, return_counts=True)
        for ind, arm in enumerate(unique_arms):
            observation_for_arm = observation_window[np.where(arm_window==arm)]
            unique_observations, obs_counts = np.unique(observation_for_arm, return_counts=True)
            p_y_temp = obs_counts / arm_counts[ind]
            new_p_y[0, int(arm), [int(x) for x in unique_observations]] = p_y_temp
            
        for i in range(new_p_y.shape[0]):
            for j in range(new_p_y.shape[1]):
                new_p_y[i,j] = new_p_y[i,j] / new_p_y[i,j].sum()
        
        #record and fix the index of this special likelihood
        self.theta_window = self.L
        self.L += 1
        self.p = np.append(self.p,1/(len(self.p)))
        self.p = self.p / self.p.sum()
        self.p_y = np.append(self.p_y,new_p_y,axis=0)
        
    def renew_window_likelihood(self):
        '''
        take place after update prior
        Step0. Fix the number of the timewindow likelihood
        Step1. synthesize a new likelihood using current belief on demand
        Step2. Update this likelihood with old likelihood
        Step3. Adjust the the demand for non-decreasing property
        Step4. Resynthesize this based on the expected demand of the adjusted likelihood
        '''
        demand = self.p @ (self.p_y @ self.obs)
        demand = np.sort(demand.reshape(1, self.K)[0])[::-1]
        est_value_pmf = (demand - np.roll(demand, shift=-1))[:-1]
        est_value_pmf = est_value_pmf / est_value_pmf.sum()
        
        #simulate under the estimated customer value and then adjust the value regarding to the direction
        window_p_y = np.zeros(shape=(1, self.K, self.N))
        for arm, price in enumerate(self.price_list):
            for simulate_round in range(100):
                customer_value = (np.random.choice(a=self.price_list_simu, size= self.N - 1, p=est_value_pmf) + 
                                  np.random.normal(0, self.price_interval, self.N - 1))
                reaction = customer_value >= price
                window_p_y[0, arm, np.sum(reaction)] += 1 / 100
        
        #count frequency from the timewindow armsequence
        arm_window = self.arm_sequence[self.t-self.window_width:self.t]
        observation_window = self.observation[self.t-self.window_width:self.t]
        unique_arms, arm_counts = np.unique(arm_window, return_counts=True)
        #insert probability value
        for ind, arm in enumerate(unique_arms):
            observation_for_arm = observation_window[np.where(arm_window==arm)]
            unique_observations, obs_counts = np.unique(observation_for_arm, return_counts=True)
            p_y_temp = obs_counts / arm_counts[ind]
            window_p_y[0, int(arm), [int(x) for x in unique_observations]] = p_y_temp
        
        #normalize for inserting time window frequency
        for i in range(window_p_y.shape[0]):
                window_p_y[i] = window_p_y[i] / window_p_y[i].sum()
        
        #Check non-increasing property
        demand_temp = window_p_y @ self.obs
        noninc_anomaly = np.sum((demand_temp - np.roll(demand_temp, -1))[:-1] < 0)
        if noninc_anomaly > 0:
            self.nonincreasing += 1
            if self.sign == 1:
                #demand increasing case, fix the anomaly by raise the demand of lower price
                for ind, element in enumerate(y):
                    demand_temp[ind] = demand_temp.max(y[ind:])
            elif self.sign == -1:
                #demand decreasing case, fix the anmoly by lower the demand of higher price
                demand_temp = demand_temp[::-1]
                for ind, element in enumerate(x):
                    demand_temp[ind] = np.min(demand_temp[ind:])
                demand_temp = demand_temp[::-1]
            else:
                demand_temp = np.sort(demand_temp)[::-1]
            est_value_pmf = (demand_temp - np.roll(demand_temp, shift=-1))[:-1]
            est_value_pmf = est_value_pmf / est_value_pmf.sum()
            
            #simulate under the estimated customer value and then adjust the value regarding to the direction
            window_p_y = np.zeros(shape=(1, self.K, self.N))
            for arm, price in enumerate(self.price_list):
                for simulate_round in range(100):
                    customer_value = (np.random.choice(a=self.price_list_simu, size= self.N - 1, p=est_value_pmf) + 
                                      np.random.normal(0, self.price_interval, self.N - 1))
                    reaction = customer_value >= price
                    window_p_y[0, arm, np.sum(reaction)] += 1 / 100
        else:
            pass
        
        #finally add 1e-4 to avoid 0 and normalize again
        window_p_y = window_p_y + 1e-6
        for i in range(window_p_y.shape[0]):
            for j in range(window_p_y.shape[1]):
                window_p_y[i,j] = window_p_y[i,j] / window_p_y[i,j].sum()

         ###Finally Insert New Window Here
        self.p_y[self.theta_window] = window_p_y
    
    def eg_shape(self):
        eg_on = np.random.choice([True, False], p=[0.1, 0.9])
        self.eg_on = eg_on
        if self.eg_on:
            detected_stamp_temp = self.detected_stamp
            self.detected_stamp = None
            print('eg triggered at ', self.t)
            sampled_arm = np.percentile(np.arange(self.K), [20, 35, 50, 65, 80]).astype(int)
            for arm in sampled_arm:
                reaction = self.update_lists(arm, self.pricing_MAB)
                self.all_posterior.append('eg_shape')
                if self.store_IDS:
                    self.IDS_results['delta'].append('eg_shape')
                    self.IDS_results['g'].append('eg_shape')
                    self.IDS_results['IR'].append('eg_shape')
            self.detected_stamp = detected_stamp_temp
            self.collection_rounds += 5
        else:
            pass
        return eg_on

    def get_pvalue(self, row):
        demand = self.p_a_y[int(row.name)] @ np.linspace(0, 10, 11) / 10
        pvalue = stats.binom_test(row['observation'], n=row['total'], p=demand)
        return pvalue
    
    def test_demand_shape(self):
        unique_sammple_size = len(np.unique(self.arm_sequence[self.detected_stamp:]))
        if unique_sammple_size < 4:
            eg_on = self.eg_shape()
            if eg_on:
                after_sample_df = pd.DataFrame({'arm':self.arm_sequence[self.detected_stamp:], 
                                        'observation':self.observation[self.detected_stamp:],
                                        'total':10})
                demand_test_df = after_sample_df.groupby('arm')[['observation', 'total']].sum()
                demand_test_df['pvalue'] = demand_test_df.apply(self.get_pvalue, axis=1)
                print(demand_test_df.pvalue)
                if any(demand_test_df.pvalue < (0.01 / demand_test_df.shape[0])):
                    #test will only be conducted once each time after detector was triggered first
                    #if tested significant -> update likelihood
                    #if tested non-significant -> keep exploitation
                    self.detected_stamp = None
                    self.eg_on = False
                    print('demand shape does not match')
                    for i in range(self.update_L):
                        self.update_likelihood()
                        self.generate_likelihood()
                else:
                    #tested non-significant, alarm goes off
                    print('tested non-significant')
                    self.detected_stamp = None
                    self.eg_on = False
            else:
                pass
        else:
            after_sample_df = pd.DataFrame({'arm':self.arm_sequence[self.detected_stamp:], 
                        'observation':self.observation[self.detected_stamp:],
                        'total':10})
            demand_test_df = after_sample_df.groupby('arm')[['observation', 'total']].sum()
            demand_test_df['pvalue'] = demand_test_df.apply(self.get_pvalue, axis=1)
            if any(demand_test_df.pvalue < (0.01 / demand_test_df.shape[0])):
                    #test will only be conducted once each time after detector was triggered first
                    #if tested significant -> update likelihood
                    #if tested non-significant -> keep exploitation
                    self.detected_stamp = None
                    self.eg_on = False
                    print('demand shape does not match')
                    for i in range(self.update_L):
                        self.update_likelihood()
                        self.generate_likelihood()
            else:
                #tested non-significant, alarm goes off
                print('tested non-significant')
                self.detected_stamp = None
                self.eg_on = False
                
    
    def detect(self):        
        #trend 1:up, 0:down
        expected_obs = (self.get_p_a_y() @ np.linspace(1, self.N-1, self.N))[int(self.arm_sequence[-1])]
        self.detector['trend'].append(self.observation[-1] > expected_obs)
        #calculation
        arm_window = self.arm_sequence[-self.window_width:]
        obs_window = self.observation[-self.window_width:]
        examine_window = obs_window[arm_window == int(self.arm_sequence[-1])]
        if len(examine_window) > 5:
            new_mean = np.mean(examine_window[-5:])
            t = len(examine_window[:-5])
            x_i = np.mean((examine_window[:-5] - new_mean) / 5)
            up = x_i + 1.7 * np.sqrt((np.log(np.log(2*t)) + (0.72 * np.log(10.4/self.alpha_cs))) / t)
            low = x_i - 1.7 * np.sqrt((np.log(np.log(2*t)) + (0.72 * np.log(10.4/self.alpha_cs))) / t)
            self.detector['upper_bound'].append(up)
            self.detector['lower_bound'].append(low)

            #testing
            if (up < 0) | (low > 0):
                if np.mean(self.detector['trend'][-20:]) > 0.5:
                    self.sign = 1
                else:
                    self.sign = -1

                #the cooler prevent the detector to be alarm repeated in short period
                if self.detected_stamp is None:
                    #will not triggered again if during near 20 rounds, but detector may tigger several time
                    #before eg_on and shape discriminator
                    self.detected_stamp = self.t
                    self.generate_likelihood()
                    print("Reward change detected at time:", self.detected_stamp)
                    print("Detected Sign:", self.sign)
                elif self.t > (self.detected_stamp + self.cool_time):
                    self.detected_stamp = self.t
                    self.generate_likelihood()
                    print("Reward change detected at time:", self.detected_stamp)
                    print("Detected Sign:", self.sign)
                else:
                    print('Detector cooling')
        else:
            pass

        
    def generate_likelihood(self):
        '''
        The generate likelihood method is called when drastic change is detected, it will estimate a customer value
        pmf by current demand estimation, and then the pmf could simulate the customer behavior.
        Furthermore, we can manipulate customer value based on this estimated customer value pmf.
        '''
        #estimate the distribution of customer value using current belief
        demand = self.p @ (self.p_y @ self.obs)
        demand = np.sort(demand.reshape(1, self.K)[0])[::-1]
        est_value_pmf = (demand - np.roll(demand, shift=-1))[:-1]
        est_value_pmf = est_value_pmf / est_value_pmf.sum()
        
        #simulate under the estimated customer value and then adjust the value regarding to the direction
        adjustment = self.price_interval * np.arange(int(-len(self.price_list)/2),int(len(self.price_list)/2))
        new_p_y = np.zeros(shape=(len(adjustment), self.K, self.N))
        for theta in range(len(adjustment)):
            for arm, price in enumerate(self.price_list):
                for simulate_round in range(1000):
                    customer_value = (np.random.choice(a=self.price_list_simu, size= self.N - 1, p=est_value_pmf) + 
                                      adjustment[theta] + np.random.normal(0, self.price_interval, self.N - 1))
                    reaction = customer_value >= price
                    new_p_y[theta, arm, np.sum(reaction)] += 1 / 1000
        
        new_p_y = new_p_y + 1e-4
        for i in range(new_p_y.shape[0]):
            for j in range(new_p_y.shape[1]):
                new_p_y[i,j] = new_p_y[i,j] / new_p_y[i,j].sum()
                
        self.p_y = np.append(self.p_y,new_p_y,axis=0)
        self.L += new_p_y.shape[0]
        self.p = np.append(self.p, np.repeat((1/self.L), new_p_y.shape[0]))
        self.p = self.p / self.p.sum()
    
    
    def get_theta_a(self):
        """
        :return: list, list of length L containing the lists of theta for which action a in [1,K] is optimal
        """
        Ta = np.zeros(shape=(self.L, self.K))
        for theta in range(self.L):
            for action in range(self.K):
                Ta[theta, action] = np.dot(self.p_y[theta,action],self.R[action])
        optimal_a_given_theta = np.array([np.argmax(x) for x in Ta])
        theta_a = []
        for action in range(self.K):
            theta_a.append(list(np.where(optimal_a_given_theta == action)[0]))
        return theta_a

    def get_pa_star(self):
        """
        :return: np.array, probabilities that action a in [1,K] is the optimal action
        """
        pa_star = np.zeros(self.K)
        for action in range(self.K):
            pa_star[action] = sum(self.p[self.theta_a[action]])
        return pa_star

    
    def get_p_a_y(self):
        """
        :return: np.array, array of shape (K,N) with probabilities of outcome Y while pulling arm A for a given prior
        """
        p_a_y = np.zeros(shape=(self.K, self.N))
        for action in range(self.K):
            p_a_y[action] = self.p.reshape(1,self.L) @ self.p_y[:,action,:]
            
        self.p_a_y = p_a_y
        return p_a_y
    
    def get_joint_ay(self):
        """
        :return: np.array, array of shape (K,K,N) with joint distribution of the outcome and the optimal arm
        while pulling arm a
        """
        p_a_star_y = np.zeros((self.K, self.K, self.N))
        for a_star in range(self.K):
            for theta in self.theta_a[a_star]:
                p_a_star_y[:, a_star, :] += self.p_y[theta] * self.p[theta]
        return p_a_star_y
            

    def get_R_star(self, p_a_star_y):
        """
        :return: float, optimal expected reward for a given prior
        """
        R_star = 0
        for action in range(self.K):
            for observation in range(self.N):
                R_star += p_a_star_y[action, action, observation] * self.R[action, observation]
        return R_star

    def get_R(self, p_a_y):
        """
        :param PY: np.array, array of shape (K,N) with probabilities of outcome Y while pulling arm A
        :return: float, expected reward for a given prior
        """
        r = np.zeros(self.K)
        for action in range(self.K):
                r[action] = np.dot(p_a_y[action, :], self.R[action])
        return r

    def get_g(self, p_a_star_y, pa_star, p_a_y):
        """
        :param joint: np.array, joint distribution P_a(y, a_star)
        :param pa: np.array, distribution of the optimal action
        :param py: np.array, probabilities of outcome Y while pulling arm A
        :return: np.array, information Gain
        """
        g = np.zeros(self.K)
        for action in range(self.K):
            for a_star in range(self.K):
                for observation in range(self.N):
                    denom = pa_star[a_star] * p_a_y[action, observation]
                    if denom == 0:
                        pass
                    else:
                        nomin = p_a_star_y[action, a_star, observation]
                        if nomin == 0:
                            pass
                        else:
                            g[action] += p_a_star_y[action, a_star, observation] * np.log(nomin / denom)
        return g

    def IR(self):
        """
        Implementation of finiteIR algorithm as defined in Russo Van Roy, p.241 algorithm 1
        :return: np.arrays, instantaneous regrets and information gains
        """
        self.p = self.p / self.p.sum()
        self.theta_a = self.get_theta_a() 
        pa_star = self.get_pa_star()
        p_a_y = self.get_p_a_y()
        p_a_star_y = self.get_joint_ay()
        R_star = self.get_R_star(p_a_star_y)
        delta = np.zeros(self.K) + R_star - self.get_R(p_a_y)
        g = self.get_g(p_a_star_y, pa_star, p_a_y)
        return delta, g

    def update_prior(self, action, reaction):
        """
        Update posterior distribution
        :param a: int, arm chose
        :param y: float, associated reward
        use self prior_lower_bound to secure all universe will be at least consider to some extent
        prior_lower_bound shrinks as the self.L goes up
        """
        data = 0 
        observation = sum(reaction)
        for theta in range(self.L):
            data += self.p[theta] * self.p_y[theta, action, observation]
            
        if data < 1e-4:
            for theta in range(self.L):
                self.p[theta] = self.p[theta] * self.p_y[theta, action, observation] / 1e-4
                self.p = self.p / self.p.sum()

        else:
            for theta in range(self.L):
                self.p[theta] = self.p[theta] * self.p_y[theta, action, observation] / data

        prior_lower_bound = 1e-08
        if sum(self.p < prior_lower_bound) > 0:
            self.p[self.p < prior_lower_bound] = prior_lower_bound
            self.p = self.p / self.p.sum()
        else:
            pass
        
        if self.window_on:
            if self.update_during_exploit:
                self.renew_window_likelihood()
            else:
                self.detect()
        else:
            pass


    def IDSAction(self, delta, g):
        """
        Since the action set in the dynamic pricing setting is ordered, it makes more sense by calculating
        only adjacent action combinations
        Implementation of IDSAction algorithm as defined in Russo & Van Roy, p. 242
        :param delta: np.array, instantaneous regrets
        :param g: np.array, information gains
        :return: int, arm to pull
        """
        Q = np.zeros(self.K-1)
        IR = np.ones(self.K-1) * np.inf
        q = np.linspace(0, 1, 1000)
        #line1
        if sum(g > 1e-04) == 0:
            arm = rd_argmax(-delta)
            if self.store_IDS:
                self.IDS_results['delta'].append(delta)
                self.IDS_results['g'].append('trivial')
                self.IDS_results['IR'].append('trivial')

        else:
            #plus a constant so that IR won't explode, supposing only trivial when all the g -> 0
            g = g + 1
            IR = delta ** 2 / g
            arm = rd_argmax(-IR)
            if self.store_IDS:
                self.IDS_results['delta'].append(delta)
                self.IDS_results['g'].append(g-1)
                self.IDS_results['IR'].append(IR)
        return arm
    
    def update_lists(self, arm, pricing_MAB):
        """
        Update all the parameters of interest after choosing the correct arm
        :param t: int, current time/round
        :param arm: int, arm chose at this round
        :param Cum_reward:  np.array, cumulative reward array up to time t-1
        :param Pull_times:  np.array, number of times arm has been pulled up to time t-1
        :param reward: np.array, rewards obtained with the policy up to time t-1
        :param arm_sequence: np.array, arm chose at each step up to time t-1
        """
        new_reward, chosen_segments_index, reaction = pricing_MAB.customer_simulation(price=self.price_list[arm])
        self.Pull_times[arm] = self.Pull_times[arm] + 1
        self.arm_sequence = np.append(self.arm_sequence, arm)
        self.observation = np.append(self.observation , sum(reaction))
        self.reward = np.append(self.reward, new_reward)
        self.Cum_reward[arm] = self.Cum_reward[arm] + new_reward
        if self.detected_stamp:
            #detected_stamp means demand shape tested is on, we can set an EG here to explore specific samples for the test if need
            try:
                if np.sum(np.dot(self.all_posterior[self.t], np.log(self.all_posterior[self.t] / self.all_posterior[self.t-1]))) < 0.001 & ~self.eg_on:
                    #only started testing if p(theta) has converged
                    self.test_demand_shape()
                else:
                    pass
            except:
                if ~self.eg_on:
                    self.test_demand_shape()
                else:
                    pass
        self.t += 1
        
        return reaction
    
    def IDS(self, T, pricing_MAB, mode='mode1', update_style='expo', base=2, p=np.nan, p_y=np.nan, R=np.nan):
        """
        Implementation of the Information Directed Sampling for Finite sets
        :param T: int, time horizon
        :return: np.arrays, reward obtained by the policy and sequence of chosen arms
        :base: it will be used as the base of the exponential function if update_style is expo, or it will be the 
        fix interval if update_style is fix
        """
        self.pricing_MAB = pricing_MAB
        initiate_rounds = self.simulate_time_initiate * self.L * self.K

        #First Prepare Likelihood
        if mode == 'mode1':
            #check if there are enough rounds for collection
            if T >= initiate_rounds:
                #run normal data collection
                #initiate p, p_y, R using first theta * action * simulation time rounds
                self.initiate_IDS_likelihood(pricing_MAB=pricing_MAB,simulate_time=self.simulate_time_initiate)
                self.collection_rounds += initiate_rounds

            else:
                sys.exit('T needs to be at least',initiate_rounds,'to run IDS_initiate, or you can put past data instead.')
        else:
            #use past p and p_y
            self.p, self.p_y, self.R = p, p_y, R
        
        #In the following rounds run IDSAction
        while self.t < T:
            
            if (self.window_width < self.t) & (~self.window_on):
                self.initiate_window_likelihood()
                self.window_on = True
                # generate once we finished collecting thetas
                self.generate_likelihood()
            else:
                pass
            
            if update_style == 'expo':
            #update on rounds that is exponentially increasing
                if (np.log(self.t) / np.log(base)) % 1 == 0:
                #doubling trick: update likelihood
                    self.update_likelihood(simulate_time=self.simulate_time_update)
                else:
                    delta, g = self.IR()
                    arm = self.IDSAction(delta, g)
                    reaction = self.update_lists(arm, pricing_MAB)
                    self.update_prior(arm, reaction)
                    self.all_posterior.append(self.p)
                    self.IDS_rounds += 1
            
            elif update_style == 'fix':
            #update on a fix interval of rounds
                if (self.t / base) % 1 == 0:
                    self.update_likelihood(simulate_time=self.simulate_time_update)
                else:
                #standard procedure
                    delta, g = self.IR()
                    arm = self.IDSAction(delta, g)
                    reaction = self.update_lists(arm, pricing_MAB)
                    self.update_prior(arm, reaction)
                    self.all_posterior.append(self.p)
                    self.IDS_rounds += 1
            
            else:
                #standard procedure
                    delta, g = self.IR()
                    arm = self.IDSAction(delta, g)
                    reaction = self.update_lists(arm, pricing_MAB)
                    self.update_prior(arm, reaction)
                    self.all_posterior.append(self.p)
                    self.IDS_rounds += 1
                
        return self.Cum_reward, self.Pull_times, self.reward, self.arm_sequence, np.array(self.all_posterior,dtype=object)

class IDS_theta_pull(IDS_pull):        
    
    def get_joint_thetay(self):
        '''
        joint distribution of theta and y given a
        '''
        p_a_theta_y = np.zeros((self.K, self.L, self.N))
        for action in range(self.K):
            for theta in range(self.L):
                p_a_theta_y[action, theta] = self.p[theta] * self.p_y[theta, action]
        return p_a_theta_y

    def get_g(self, p_a_theta_y, p_a_y):
        """
        :param joint: np.array, joint distribution P_a(y, a_star)
        :param pa: np.array, distribution of the optimal action
        :param py: np.array, probabilities of outcome Y while pulling arm A
        :return: np.array, information Gain
        """
        g = np.zeros(self.K)
        for action in range(self.K):
            for theta in range(self.L):
                for observation in range(self.N):
                    denom = self.p[theta] * p_a_y[action, observation]
                    if denom == 0:
                        pass
                    else:
                        nomin = p_a_theta_y[action, theta, observation]
                        if nomin == 0:
                            pass
                        else:
                            g[action] += nomin * np.log(nomin / denom)
        return g

    def IR(self):
        #一次算完algo1 得到finite的delta跟info gain
        """
        Implementation of finiteIR algorithm as defined in Russo Van Roy, p.241 algorithm 1
        :return: np.arrays, instantaneous regrets and information gains
        """
        self.p = self.p / self.p.sum()
        self.theta_a = self.get_theta_a() 
        pa_star = self.get_pa_star()
        p_a_y = self.get_p_a_y()
        p_a_star_y = self.get_joint_ay()
        p_a_theta_y = self.get_joint_thetay()
        R_star = self.get_R_star(p_a_star_y)
        delta = np.zeros(self.K) + R_star - self.get_R(p_a_y)
        g = self.get_g(p_a_theta_y, p_a_y)
        return delta, g