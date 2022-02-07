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

def draw_exp_graph(IDS_1, IDS_2, IDS_3, IDST_1, UCB_1, UCB_2, UCBT, UCBPI, TS, EG_1, EG_2, EG_3, title):
    
    IDS_1_mean = IDS_1.mean()
    IDS_2_mean = IDS_2.mean()
    IDS_3_mean = IDS_3.mean()
    IDST_1_mean = IDST_1.mean()
    UCB_1_mean = UCB_1.mean()
    UCB_2_mean =UCB_2.mean()
    UCBT_mean = UCBT.mean()
    UCBPI_mean = UCBPI.mean()
    TS_mean = TS.mean()
    EG_1_mean = EG_1.mean()
    EG_2_mean = EG_2.mean()
    EG_3_mean = EG_3.mean()
    
    T = np.arange(EG_1.shape[1])
    trial = EG_1.shape[0]
    
    plt.figure(figsize=(30,20))
    sns.set(font_scale=2, style='white')
    
    plt.plot(EG_1_mean,alpha=1,color='#000000', linestyle='-')
    plt.fill_between(T, EG_1.min(), EG_1.max(), color='#000000', alpha=0.1)
    
    plt.plot(EG_2_mean,alpha=1,color='#000000', linestyle='-.')
    plt.fill_between(T, EG_2.min(), EG_2.max(), color='#000000', alpha=0.1)
    
    plt.plot(EG_3_mean,alpha=1,color='#000000', linestyle='--')
    plt.fill_between(T, EG_3.min(), EG_3.max(), color='#000000', alpha=0.1)
    
    plt.plot(TS_mean,alpha=1,color='#666666', linestyle='-')
    plt.fill_between(T, TS.min(), TS.max(), color='#666666', alpha=0.2)
    
    plt.plot(UCB_1_mean,alpha=1,color='#666600', linestyle='-')
    plt.fill_between(T, UCB_1.min(), UCB_1.max(), color='#666600', alpha=0.2)
    
    plt.plot(UCB_2_mean,alpha=1,color='#666600', linestyle='-.')
    plt.fill_between(T, UCB_2.min(), UCB_2.max(), color='#666600', alpha=0.2)
    
    plt.plot(UCBT_mean,alpha=1,color='#666600', linestyle='--')
    plt.fill_between(T, UCBT.min(), UCBT.max(), color='#666600', alpha=0.2)

    plt.plot(UCBPI_mean,alpha=1,color='#666600', linestyle=':')
    plt.fill_between(T, UCBPI.min(), UCBPI.max(), color='#666600', alpha=0.2)

    plt.plot(IDS_1_mean,alpha=1,color='#FF6666', linestyle='-')
    plt.fill_between(T, IDS_1.min(), IDS_1.max(), color='#FF6666', alpha=0.2)

    plt.plot(IDS_2_mean,alpha=1,color='#FF6666', linestyle='-.')
    plt.fill_between(T, IDS_2.min(), IDS_2.max(), color='#FF6666', alpha=0.2)

    plt.plot(IDS_3_mean,alpha=1,color='#FF6666', linestyle='--')
    plt.fill_between(T, IDS_3.min(), IDS_3.max(), color='#FF6666', alpha=0.2)

    plt.plot(IDST_1_mean,alpha=1,color='#FF6666', linestyle=':')
    plt.fill_between(T, IDST_1.min(), IDST_1.max(), color='#FF6666', alpha=0.2)
    
    plt.legend(['EG_1', 'EG_2', 'EG_3', 'TS', 'UCB1', 'UCB2', 'UCB-tuned', 'UCBPI', 'IDS_1 L=2, n=1', 'IDS2 L=4, n=1', 'IDS L=2, n=2', 'IDS Theta'],
              bbox_to_anchor=(0.85, -0.05), ncol=6, fancybox=True)
    plt.title(title)
    plt.xlabel('Horizon')
    plt.ylabel('Cumulative Reward')
    plt.show()

def draw_arm(IDS, TS, UCB, EG):
    sns.set(font_scale=1.5, style='white')
    figure, axis = plt.subplots(2, 2, figsize=(20,12))

    trial = EG.shape[0]
    X = np.arange(EG.shape[1])
    for i in range(trial):
        axis[0, 0].scatter(x=X, y=IDS.iloc[i], alpha=0.1, s=10, color='#FF6666')
    axis[0, 0].set_ylim([-1, 20])
    axis[0, 0].set_ylabel('Pulled Arms')
    axis[0, 0].set_title('IDS')

    for i in range(trial):
        axis[0, 1].scatter(x=X, y=TS.iloc[i], alpha=0.1, s=10, color='#666666')
    axis[0, 1].set_ylim([-1, 20])
    axis[0, 1].set_ylabel('Pulled Arms')
    axis[0, 1].set_title('TS')

    for i in range(trial):
        axis[1, 0].scatter(x=X, y=UCB.iloc[i], alpha=0.1, s=10, color='#666600')
    axis[1, 0].set_ylim([-1, 20])
    axis[1, 0].set_ylabel('Pulled Arms')
    axis[1, 0].set_title('UCB')

    for i in range(trial):
        axis[1, 1].scatter(x=X, y=EG.iloc[i], alpha=0.1, s=10, color='#996633')
    axis[1, 1].set_ylim([-1, 20])
    axis[1, 1].set_ylabel('Pulled Arms')
    axis[1, 1].set_title('EG')

    figure.text(0.5, 0.05, 'Horizon')
    plt.show()