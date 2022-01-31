import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from scipy import stats

def rd_argmax(vector):
    """
    Compute random among eligible maximum indices
    :param vector: np.array
    :return: int, random index among eligible maximum indices
    """
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return np.random.choice(indices)

def draw_empirical_likelihood(price_MAB, price_list,simu=1000):
    df = pd.DataFrame(columns=['arm','price','reward'])
    
    for arm in range(price_list.shape[0]):
        observation = []
        all_p_y = np.zeros(shape=(11))
        for rounds in range(simu):
            reward, _, reaction = price_MAB.customer_simulation(price_list[arm])
            df = df.append({'arm':arm,'price':price_list[arm],'reward':reward},ignore_index=True)
            observation.append(sum(reaction))
        for i in range(11):
            all_p_y[i] = observation.count(i) / simu
        #plot1: empirical likelihood
        sns.set(font_scale=1)
        sns.barplot(x=np.arange(11),y=all_p_y)
        plt.title('the empirical likelihood of observation for arm: '+str(arm)+' price: '+str(price_list[arm]))
        plt.show()
        
    #plot2: kde estimation of all action
    sns.set(font_scale=4)
    sns.set_style('whitegrid')
    plt.figure(figsize=(50,30))
    sns.kdeplot(x='reward', data=df, hue='price',palette='Set2')
    plt.title('the reward distribution for prices')
    plt.xlim([-5,5])
    plt.show()
    
    #plot3: mean reward
    sns.set(font_scale=1.5)
    plt.figure(figsize=(30,20))
    plt.plot(df.groupby('arm')['reward'].mean())
    plt.title('mean reward over actions')
    plt.show()
    optimal_arm = np.argmax(df.groupby('price')['reward'].mean())
    optimal_price = price_list[optimal_arm]
    print('optimal arm:', optimal_arm, 'optimal_price', optimal_price)
    return df

def exp_3_best_arm(price_MAB, price_list,simu=1000):
    
    df = pd.DataFrame(columns=['arm','price','reward'])
    
    for arm in range(price_list.shape[0]):
        for rounds in range(simu):
            reward, _, reaction = price_MAB.customer_simulation(price_list[arm])
            df = df.append({'arm':arm,'price':price_list[arm],'reward':reward},ignore_index=True)

    sns.set(font_scale=1.5)
    plt.figure(figsize=(30,20))
    plt.plot(df.groupby('arm')['reward'].mean())
    plt.title('mean reward over actions')
    plt.show()
    optimal_arm = np.argmax(df.groupby('price')['reward'].mean())
    optimal_price = price_list[optimal_arm]
    print('optimal arm:', optimal_arm, 'optimal_price', optimal_price)
    return df

def draw_exp_graph(IDS,UCB,sd=2):
    IDS_mean = IDS.mean()
    IDS_std = IDS.std()
    UCB_mean = UCB.mean()
    UCB_std = UCB.std()
    T = np.arange(IDS.shape[1])
    
    plt.figure(figsize=(30,20))
    
    for i in range(UCB.shape[0]):
        plt.plot(UCB.iloc[i],alpha=0.4,color='gray')
    plt.plot(UCB_mean,alpha=1,color='black')
    plt.fill_between(T, UCB_mean-sd*UCB_std, UCB_mean+sd*UCB_std, color='gray', alpha=0.2)

    for i in range(IDS.shape[0]):
        plt.plot(IDS.iloc[i],alpha=0.4,color='pink')
    plt.plot(IDS_mean,alpha=1,color='red')
    plt.fill_between(T, IDS_mean-sd*IDS_std, IDS_mean+sd*IDS_std, color='pink', alpha=0.2)

    plt.show()

def harmonic_KL(vector1, vector2):
    KL_1 = np.sum(np.dot(vector1, np.log(vector1 / vector2)))
    KL_2 = np.sum(np.dot(vector2, np.log(vector2 / vector1)))
    harmonic = 1 / (1/KL_1 + 1/KL_2)
    return harmonic
