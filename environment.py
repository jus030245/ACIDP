import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy import stats

class brownian():
    """
    A Brownian motion class constructor
    """
    def __init__(self,x0=0):
        """
        Init class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
        self.step = 1
        self.motion = []
    
    def gen_onestep(self, T=1000):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments:
            n_step: Number of steps
            
        Returns:
            A NumPy array with `n_steps` points
        """
        yi = np.random.normal(0, 1)
        # Weiner process
        self.x0 = self.x0 + (yi/np.sqrt(T))
        self.motion.append(self.x0)
        self.step += 1
        
        return self.x0
    
    def gen_sequence(self, step=1000):
        """
        Generate the whole process given steps
        """
        self.motion = []
        self.x0 = 0
        self.step = 1
        for i in range(step):
            self.gen_onestep(step)
        return self.motion



### Customer Simulation
class pricing_MAB():
    def __init__(self, price_list=np.linspace(0.01,1,100), segments=1000,update_freq=10, exp='exp3', T=1000):
        '''
        R is the reward function that can map the observation to the final reward
        
        In the exp4 design, there will be 1% of the segment are set of high-risk customers.
        And they has a default rate of 10%.
        Once triggered, -10 reward will be given
        
        '''
        self.price_list = price_list
        self.segments = segments
        self.update_freq = update_freq
        self.exp = exp
        self.season = 0
#         self.R = self.price_list.reshape(K,1) @ np.arange(N).reshape(1,N)
        if exp == 'exp6':
            self.t = 0
            self.brownian = brownian(0)
            self.brownian_sequence = self.brownian.gen_sequence(T)
            

    def segments_means(self, param1=3,param2=6, shift=0):
        #default from beta(3,6)
        #means = True or list
        if self.exp in ['exp3','exp5','exp6']:
            self.seg_means = np.random.beta(param1,param2,self.segments) + shift
            if self.segments < 30:
                print('segments means',self.seg_means)
            else:
                print('first 30 segments means',self.seg_means[:30])
                
        elif self.exp == 'exp4':
            self.seg_means = np.random.beta(param1,param2,self.segments)
            if self.segments < 30:
                print('segments means',self.seg_means)
            else:
                print('first 30 segments means',self.seg_means[:30])
            self.high_risk_semgents = np.random.choice(self.segments, round(self.segments * 0.01), replace=False)
            
        else:
            print('need to design new means')
        
            
    def customer_simulation(self, price, within=True):
        if self.exp == 'exp3':
            chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
            chosen_segments = self.seg_means[chosen_segments_index]
            customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq)
            reaction = (customer_values >= price)
            reward = price * np.sum(reaction)
        elif self.exp == 'exp4':
            chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
            chosen_segments = self.seg_means[chosen_segments_index]
            customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq)
            reaction = (customer_values >= price)
            #count number of high_risk_semgents
            count = 0
            for segment in chosen_segments_index[reaction]: #only deal segments will trigger default -> so the price should not be set too low
                if segment in self.high_risk_semgents:
                    count += 1
            penalty = sum(np.random.binomial(1, 0.1, count)) * -10
            reward = price * np.sum(reaction) + penalty
        elif self.exp == 'exp5':
            chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
            chosen_segments = self.seg_means[chosen_segments_index]
            #set a constant 0.2 so that the sin function is bound by [-0.3,0.3]
            customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq) + 0.3 * np.sin(self.season)
            reaction = (customer_values >= price)
            reward = price * np.sum(reaction)
            #update pi/4000 once at a time such that 8000 rounds will fulfill a complete cycle of sin from -1 to 1
            self.season += (np.pi / 4000)
        elif self.exp == 'exp6':
            chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
            chosen_segments = self.seg_means[chosen_segments_index]
            customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq) + self.brownian_sequence[self.t]
            reaction = (customer_values >= price)
            reward = price * np.sum(reaction)
            self.t += 1
        else:
            print('need to design new simulation')
            
        return reward, chosen_segments_index, reaction
    
    def reset_seasonality(self):
        self.season = 0
        
    def reset_volatility(self):
        self.t = 0