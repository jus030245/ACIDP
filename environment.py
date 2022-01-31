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
        self.t = 0
        self.T = T
        if exp == 'exp6':
            self.brownian = brownian(0)
            self.brownian_sequence = self.brownian.gen_sequence(T)
            

    def segments_means(self, param1=3,param2=6, shift=0.3):
        if self.exp in ['exp1', 'exp4', 'exp5']:
            self.seg_means = np.random.beta(param1,param2,self.segments)
            
        elif self.exp == 'exp2':
            self.seg_means = np.random.beta(param1,param2,self.segments)
            self.seg_means_2 = np.random.beta(param1,param2,self.segments) + shift

        elif self.exp == 'exp3':
            self.seg_means = np.random.beta(param1,param2,self.segments)
            self.seg_means_2 = np.random.beta(param1,param2,self.segments) - shift
        
        elif self.exp == 'exp6':
            self.seg_means = np.random.beta(param1,param2,self.segments)
            self.seg_means_2 = np.random.beta(param2,param1,self.segments)
                    
    def customer_simulation(self, price, within=True):
        if self.exp == 'exp1':
            chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
            chosen_segments = self.seg_means[chosen_segments_index]
            customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq)
            reaction = (customer_values >= price)
            reward = price * np.sum(reaction)
        
        elif self.exp in ['exp2', 'exp3', 'exp6']:
            stage_1 = int(self.T / 2)
            if self.t < stage_1:
                chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
                chosen_segments = self.seg_means[chosen_segments_index]
                customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq)
                reaction = (customer_values >= price)
                reward = price * np.sum(reaction)
                self.t += 1
            else:
                chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
                chosen_segments = self.seg_means_2[chosen_segments_index]
                customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq)
                reaction = (customer_values >= price)
                reward = price * np.sum(reaction)
                self.t += 1
            if self.t == self.T:
                self.reset_t()
    
        elif self.exp == 'exp4':
            chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
            chosen_segments = self.seg_means[chosen_segments_index]
            #set a constant 0.3 so that the sin function is bound by [-0.3,0.3]
            customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq) + 0.3 * np.sin(self.season)
            reaction = (customer_values >= price)
            reward = price * np.sum(reaction)
            #update pi/(1/T/2) once at a time such that 8000 rounds will fulfill a complete cycle of sin from -1 to 1
            #complete two season in the experiment
            self.t += (np.pi / (self.T/2))
            if self.t == (self.T - 1):
                self.reset_t()

        elif self.exp == 'exp5':
            chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
            chosen_segments = self.seg_means[chosen_segments_index]
            customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq) + self.brownian_sequence[self.t]
            reaction = (customer_values >= price)
            reward = price * np.sum(reaction)
            self.t += 1
            if self.t == (self.T - 1):
                self.reset_t()
        return reward, chosen_segments_index, reaction
        
    def reset_t(self):
        self.t = 0

    def true_optimal(self):
        if self.exp == 'exp1':
            df, optimal_arm = self.simulator_exp1()
            print("The true optimal arm is:", optimal_arm)
            return df
            
        if self.exp in ['exp2', 'exp3', 'exp6']:
            df, optimal_arm, df_2, optimal_arm_2  = self.simulator_exp236()
            print("The true optimal arm in first", int(self.T/2), "rounds is:", optimal_arm)
            print("The true optimal arm in the rest is:", optimal_arm_2)
            return df, df_2

        if self.exp == ['exp4', 'exp5']:
            df, optimal_arm, df_2, optimal_arm_2  = self.simulator_exp45()
            print("The highest true optimal is", optimal_arm)
            print("The lowest true optimal is:", optimal_arm_2)
            return df, df_2


    def simulator_exp1(self, simu=1000):
        df = pd.DataFrame(columns=['arm','price','reward'])
        for arm in range(self.price_list.shape[0]):
            price = self.price_list[arm]
            observation = []
            all_p_y = np.zeros(shape=(11))
            for rounds in range(simu):
                reward, _, reaction = self.customer_simulation(self.price_list[arm])
                df = df.append({'arm':arm,'price':price, 'reward':reward},ignore_index=True)
                observation.append(sum(reaction))
            for i in range(11):
                all_p_y[i] = observation.count(i) / simu
        optimal_arm = np.argmax(df.groupby('price')['reward'].mean())
        return df, optimal_arm

    def simulator_exp236(self, simu=1000):
        df = pd.DataFrame(columns=['arm','price','reward'])
        for arm in range(self.price_list.shape[0]):
            price = self.price_list[arm]
            observation = []
            all_p_y = np.zeros(shape=(11))
            for rounds in range(simu):
                chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
                chosen_segments = self.seg_means[chosen_segments_index]
                customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq)
                reaction = (customer_values >= price)
                reward = price * np.sum(reaction)
                df = df.append({'arm':arm,'price':price,'reward':reward},ignore_index=True)
                observation.append(sum(reaction))
            for i in range(11):
                all_p_y[i] = observation.count(i) / simu
        optimal_arm = np.argmax(df.groupby('price')['reward'].mean())

        df_2 = pd.DataFrame(columns=['arm','price','reward'])
        for arm in range(self.price_list.shape[0]):
            price = self.price_list[arm]
            observation = []
            all_p_y = np.zeros(shape=(11))
            for rounds in range(simu):
                chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
                chosen_segments = self.seg_means_2[chosen_segments_index]
                customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq)
                reaction = (customer_values >= price)
                reward = price * np.sum(reaction)
                df_2 = df_2.append({'arm':arm,'price':price,'reward':reward},ignore_index=True)
                observation.append(sum(reaction))
            for i in range(11):
                all_p_y[i] = observation.count(i) / simu
        optimal_arm_2 = np.argmax(df_2.groupby('price')['reward'].mean())
            
        return df, optimal_arm, df_2, optimal_arm_2

        
    def simulator_exp45(self, simu=1000):
        if self.exp == 'exp4':
            highest = 0.3
            lowest = -0.3
        elif self.exp == 'exp5':
            highest = max(self.brownian_sequence)
            lowest = min(self.brownian_sequence)

        df = pd.DataFrame(columns=['arm','price','reward'])
        for arm in range(self.price_list.shape[0]):
            price = self.price_list[arm]
            observation = []
            all_p_y = np.zeros(shape=(11))
            for rounds in range(simu):
                chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
                chosen_segments = self.seg_means[chosen_segments_index]
                customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq) + highest
                reaction = (customer_values >= price)
                reward = price * np.sum(reaction)
                df = df.append({'arm':arm,'price':price,'reward':reward},ignore_index=True)
                observation.append(sum(reaction))
            for i in range(11):
                all_p_y[i] = observation.count(i) / simu
        optimal_arm = np.argmax(df.groupby('price')['reward'].mean())

        df_2 = pd.DataFrame(columns=['arm','price','reward'])
        for arm in range(self.price_list.shape[0]):
            price = self.price_list[arm]
            observation = []
            all_p_y = np.zeros(shape=(11))
            for rounds in range(simu):
                chosen_segments_index = np.random.choice(self.segments, self.update_freq, replace=True)
                chosen_segments = self.seg_means_2[chosen_segments_index]
                customer_values = chosen_segments + np.random.normal(0, 0.1, self.update_freq) + lowest
                reaction = (customer_values >= price)
                reward = price * np.sum(reaction)
                df_2 = df_2.append({'arm':arm,'price':price,'reward':reward},ignore_index=True)
                observation.append(sum(reaction))
            for i in range(11):
                all_p_y[i] = observation.count(i) / simu
        optimal_arm_2 = np.argmax(df_2.groupby('price')['reward'].mean())
            
        return df, optimal_arm, df_2, optimal_arm_2
