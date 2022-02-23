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
    plt.fill_between(T, EG_1.min(), EG_1.max(), color='#000000', alpha=0.2)
    
    plt.plot(EG_2_mean,alpha=1,color='#000000', linestyle='-.')
    plt.fill_between(T, EG_2.min(), EG_2.max(), color='#000000', alpha=0.2)
    
    plt.plot(EG_3_mean,alpha=1,color='#000000', linestyle='--')
    plt.fill_between(T, EG_3.min(), EG_3.max(), color='#000000', alpha=0.2)
    
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
    axis[0, 0].set_title('ACIDP')

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

def generator_demo(IDS, t, size=10000):
    try:
        L = IDS.all_posterior[t].size
        posterior = IDS.all_posterior[t][:L]
        q = IDS.p_y[:L]
        p_a_y = np.zeros(shape=(IDS.K, IDS.N))
        for action in range(IDS.K):
            p_a_y[action] = posterior.reshape(1, L) @ q[:,action,:]
        demand = p_a_y @ np.linspace(0, IDS.N-1, IDS.N) / 10
        demand = np.sort(demand.reshape(1, IDS.K)[0])[::-1]
        est_value_pmf = (demand - np.roll(demand, shift=-1))[:-1]
        est_value_pmf = est_value_pmf / est_value_pmf.sum()
        customer_value = (np.random.choice(a=IDS.price_list_simu, size= size, p=est_value_pmf) + 
                        np.random.normal(0, IDS.price_interval, size))
    except:
        print('Chosen T Is Under Exploration')
    return demand, est_value_pmf, customer_value


def get_low(optimal_reward, df):
    return (optimal_reward - df.max()).values[0]
def get_high(optimal_reward, df):
    return (optimal_reward - df.min()).values[0]

def draw_regret(optimal_reward, IDS_1, IDS_2, IDS_3, IDST_1, UCB_1, UCB_2, UCBT, UCBPI, TS, EG_1, EG_2, EG_3, title, bounds=True):
    
    IDS_1_mean = (optimal_reward - IDS_1.mean()).values[0]
    IDS_2_mean = (optimal_reward - IDS_2.mean()).values[0]
    IDS_3_mean = (optimal_reward - IDS_3.mean()).values[0]
    IDST_1_mean = (optimal_reward - IDST_1.mean()).values[0]
    UCB_1_mean = (optimal_reward - UCB_1.mean()).values[0]
    UCB_2_mean = (optimal_reward - UCB_2.mean()).values[0]
    UCBT_mean = (optimal_reward - UCBT.mean()).values[0]
    UCBPI_mean = (optimal_reward - UCBPI.mean()).values[0]
    TS_mean = (optimal_reward - TS.mean()).values[0]
    EG_1_mean = (optimal_reward - EG_1.mean()).values[0]
    EG_2_mean = (optimal_reward - EG_2.mean()).values[0]
    EG_3_mean = (optimal_reward - EG_3.mean()).values[0]
    
    T = np.arange(EG_1.shape[1])
    trial = EG_1.shape[0]
    
    plt.figure(figsize=(30,20))
    sns.set(font_scale=2, style='white')
    
    plt.plot(EG_1_mean,alpha=1,color='#000000', linestyle='-')
    plt.plot(EG_2_mean,alpha=1,color='#000000', linestyle='-.')    
    plt.plot(EG_3_mean,alpha=1,color='#000000', linestyle='--')
    
    plt.plot(TS_mean,alpha=1,color='#666666', linestyle='-')
    
    plt.plot(UCB_1_mean,alpha=1,color='#666600', linestyle='-')
    plt.plot(UCB_2_mean,alpha=1,color='#666600', linestyle='-.')
    plt.plot(UCBT_mean,alpha=1,color='#666600', linestyle='--')
    plt.plot(UCBPI_mean,alpha=1,color='#666600', linestyle=':')


    plt.plot(IDS_1_mean,alpha=1,color='#FF6666', linestyle='-')
    plt.plot(IDS_2_mean,alpha=1,color='#FF6666', linestyle='-.')
    plt.plot(IDS_3_mean,alpha=1,color='#FF6666', linestyle='--')
    plt.plot(IDST_1_mean,alpha=1,color='#FF6666', linestyle=':')

    
    if bounds:
        plt.fill_between(T, get_low(optimal_reward, EG_1), get_high(optimal_reward, EG_1), color='#000000', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, EG_2), get_high(optimal_reward, EG_2), color='#000000', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, EG_3), get_high(optimal_reward, EG_3), color='#000000', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, TS), get_high(optimal_reward, TS), color='#666666', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, UCB_1), get_high(optimal_reward, UCB_1), color='#666600', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, UCB_2), get_high(optimal_reward, UCB_2), color='#666600', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, UCBT), get_high(optimal_reward, UCBT), color='#666600', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, UCBPI), get_high(optimal_reward, UCBPI), color='#666600', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, IDS_1), get_high(optimal_reward, IDS_1), color='#FF6666', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, IDS_2), get_high(optimal_reward, IDS_2), color='#FF6666', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, IDS_3), get_high(optimal_reward, IDS_3), color='#FF6666', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, IDST_1), get_high(optimal_reward, IDST_1), color='#FF6666', alpha=0.1)

        
        
        
    plt.legend(['EG $ε$=0.05', 'EG $ε$=0.1', 'EG $ε$-0.15', 'TS', 'UCB c=1', 'UCB c=2', 'UCB-tuned',
                'UCBPI', 'ACIDP L=2, n=1', 'ACIDP L=4, n=1', 'ACIDP L=2, n=2',
                'ACIDP-$θ$ L=2, n=1'],
              bbox_to_anchor=(0.9, -0.05), ncol=6, fancybox=True)
    plt.title(title)
    plt.xlabel('Horizon')
    plt.ylabel('Cumulative Regret')
    plt.show()

def draw_regret_6(optimal_reward, IDS_1, IDS_2, IDS_3, IDST_1, IDSN, IDSW, UCB_1, UCB_2, UCBT, UCBPI, TS, EG_1, EG_2, EG_3, title, bounds=True):
    
    IDS_1_mean = (optimal_reward - IDS_1.mean()).values[0]
    IDS_2_mean = (optimal_reward - IDS_2.mean()).values[0]
    IDS_3_mean = (optimal_reward - IDS_3.mean()).values[0]
    IDST_1_mean = (optimal_reward - IDST_1.mean()).values[0]
    IDSN_mean = (optimal_reward - IDSN.mean()).values[0]
    IDSW_mean = (optimal_reward - IDSW.mean()).values[0]
    UCB_1_mean = (optimal_reward - UCB_1.mean()).values[0]
    UCB_2_mean = (optimal_reward - UCB_2.mean()).values[0]
    UCBT_mean = (optimal_reward - UCBT.mean()).values[0]
    UCBPI_mean = (optimal_reward - UCBPI.mean()).values[0]
    TS_mean = (optimal_reward - TS.mean()).values[0]
    EG_1_mean = (optimal_reward - EG_1.mean()).values[0]
    EG_2_mean = (optimal_reward - EG_2.mean()).values[0]
    EG_3_mean = (optimal_reward - EG_3.mean()).values[0]
    
    T = np.arange(EG_1.shape[1])
    trial = EG_1.shape[0]
    
    plt.figure(figsize=(30,20))
    sns.set(font_scale=2, style='white')
    
    plt.plot(EG_1_mean,alpha=1,color='#000000', linestyle='-')
    plt.plot(EG_2_mean,alpha=1,color='#000000', linestyle='-.')    
    plt.plot(EG_3_mean,alpha=1,color='#000000', linestyle='--')
    
    plt.plot(TS_mean,alpha=1,color='#666666', linestyle='-')
    
    plt.plot(UCB_1_mean,alpha=1,color='#666600', linestyle='-')
    plt.plot(UCB_2_mean,alpha=1,color='#666600', linestyle='-.')
    plt.plot(UCBT_mean,alpha=1,color='#666600', linestyle='--')
    plt.plot(UCBPI_mean,alpha=1,color='#666600', linestyle=':')


    plt.plot(IDS_1_mean,alpha=1,color='#FF6666', linestyle='-')
    plt.plot(IDS_2_mean,alpha=1,color='#FF6666', linestyle='-.')
    plt.plot(IDS_3_mean,alpha=1,color='#FF6666', linestyle='--')
    plt.plot(IDST_1_mean,alpha=1,color='#FF6666', linestyle=':')
    plt.plot(IDSN_mean,alpha=1,color='#0000FF', linestyle='-')
    plt.plot(IDSW_mean,alpha=1,color='#FFA500', linestyle='-')
    
    if bounds:
        plt.fill_between(T, get_low(optimal_reward, EG_1), get_high(optimal_reward, EG_1), color='#000000', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, EG_2), get_high(optimal_reward, EG_2), color='#000000', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, EG_3), get_high(optimal_reward, EG_3), color='#000000', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, TS), get_high(optimal_reward, TS), color='#666666', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, UCB_1), get_high(optimal_reward, UCB_1), color='#666600', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, UCB_2), get_high(optimal_reward, UCB_2), color='#666600', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, UCBT), get_high(optimal_reward, UCBT), color='#666600', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, UCBPI), get_high(optimal_reward, UCBPI), color='#666600', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, IDS_1), get_high(optimal_reward, IDS_1), color='#FF6666', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, IDS_2), get_high(optimal_reward, IDS_2), color='#FF6666', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, IDS_3), get_high(optimal_reward, IDS_3), color='#FF6666', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, IDST_1), get_high(optimal_reward, IDST_1), color='#FF6666', alpha=0.1)
        plt.fill_between(T, get_low(optimal_reward, IDSN), get_high(optimal_reward, IDSN), color='#0000FF', alpha=0.05)
        plt.fill_between(T, get_low(optimal_reward, IDSW), get_high(optimal_reward, IDSW), color='#FFA500', alpha=0.1)
        
        
    plt.legend(['EG $ε$=0.05', 'EG $ε$=0.1', 'EG $ε$-0.15', 'TS', 'UCB c=1', 'UCB c=2', 'UCB-tuned',
                'UCBPI', 'ACIDP L=2, n=1', 'ACIDP L=4, n=1', 'ACIDP L=2, n=2',
                'ACIDP-$θ$ L=2, n=1', 'ACIDP-without Transferability Test L=2, n=1', 'ACIDP-Variant L=2, n=1'],
              bbox_to_anchor=(1, -0.05), ncol=7, fancybox=True)
    plt.title(title)
    plt.xlabel('Horizon')
    plt.ylabel('Cumulative Regret')
    plt.show()

def draw_arm_withopt(IDS, TS, UCB, EG, optimal_arm):
    sns.set(font_scale=1.5, style='white')
    figure, axis = plt.subplots(2, 2, figsize=(20,12))

    trial = EG.shape[0]
    X = np.arange(EG.shape[1])
    for i in range(trial):
        axis[0, 0].scatter(x=X, y=IDS.iloc[i], alpha=0.1, s=10, color='#FF6666')
    axis[0, 0].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[0, 0].set_ylim([-1, 20])
    axis[0, 0].set_ylabel('Pulled Arms')
    axis[0, 0].set_title('ACIDP')

    for i in range(trial):
        axis[0, 1].scatter(x=X, y=TS.iloc[i], alpha=0.1, s=10, color='#666666')
    axis[0, 1].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[0, 1].set_ylim([-1, 20])
    axis[0, 1].set_ylabel('Pulled Arms')
    axis[0, 1].set_title('TS')

    for i in range(trial):
        axis[1, 0].scatter(x=X, y=UCB.iloc[i], alpha=0.1, s=10, color='#666600')
    axis[1, 0].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[1, 0].set_ylim([-1, 20])
    axis[1, 0].set_ylabel('Pulled Arms')
    axis[1, 0].set_title('UCB')

    for i in range(trial):
        axis[1, 1].scatter(x=X, y=EG.iloc[i], alpha=0.1, s=10, color='#996633')
    axis[1, 1].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[1, 1].set_ylim([-1, 20])
    axis[1, 1].set_ylabel('Pulled Arms')
    axis[1, 1].set_title('EG')

    figure.text(0.5, 0.05, 'Horizon')
    plt.show()

def draw_arm_withopt_6(IDSN, IDSW, IDS, TS, UCB, EG, optimal_arm):
    sns.set(font_scale=1.5, style='white')
    figure, axis = plt.subplots(2, 3, figsize=(20,12))

    trial = EG.shape[0]
    X = np.arange(EG.shape[1])
    for i in range(trial):
        axis[0, 0].scatter(x=X, y=IDSN.iloc[i], alpha=0.1, s=10, color='#0000FF')
    axis[0, 0].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[0, 0].set_ylim([-1, 20])
    axis[0, 0].set_ylabel('Pulled Arms')
    axis[0, 0].set_title('ACIDP-Without Transferability Test')
    
    for i in range(trial):
        axis[0, 1].scatter(x=X, y=IDS.iloc[i], alpha=0.1, s=10, color='#FF6666')
    axis[0, 1].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[0, 1].set_ylim([-1, 20])
    axis[0, 1].set_ylabel('Pulled Arms')
    axis[0, 1].set_title('ACIDP')
    
    for i in range(trial):
        axis[0, 2].scatter(x=X, y=IDSW.iloc[i], alpha=0.1, s=10, color='#FFA500')
    axis[0, 2].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[0, 2].set_ylim([-1, 20])
    axis[0, 2].set_ylabel('Pulled Arms')
    axis[0, 2].set_title('ACIDP-Variant')

    for i in range(trial):
        axis[1, 0].scatter(x=X, y=TS.iloc[i], alpha=0.1, s=10, color='#666666')
    axis[1, 0].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[1, 0].set_ylim([-1, 20])
    axis[1, 0].set_ylabel('Pulled Arms')
    axis[1, 0].set_title('TS')

    for i in range(trial):
        axis[1, 1].scatter(x=X, y=UCB.iloc[i], alpha=0.1, s=10, color='#666600')
    axis[1, 0].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[1, 1].set_ylim([-1, 20])
    axis[1, 1].set_ylabel('Pulled Arms')
    axis[1, 1].set_title('UCB')

    for i in range(trial):
        axis[1, 2].scatter(x=X, y=EG.iloc[i], alpha=0.1, s=10, color='#996633')
    axis[1, 2].plot(optimal_arm, color='r', linewidth=3, linestyle='--')
    axis[1, 2].set_ylim([-1, 20])
    axis[1, 2].set_ylabel('Pulled Arms')
    axis[1, 2].set_title('EG')

    figure.text(0.5, 0.05, 'Horizon')
    plt.show()

# Color change version
# def draw_regret(optimal_reward, groups, title, bounds=True):
    
#     IDS_1_mean = (optimal_reward - groups.loc['IDS_1'].values).values[0]
#     IDS_2_mean = (optimal_reward - groups.loc['IDS_2'].values).values[0]
#     IDS_3_mean = (optimal_reward - groups.loc['IDS_3'].values).values[0]
#     IDST_1_mean = (optimal_reward - groups.loc['IDST_1'].values).values[0]
#     UCB_1_mean = (optimal_reward - groups.loc['UCB_1'].values).values[0]
#     UCB_2_mean = (optimal_reward - groups.loc['UCB_2'].values).values[0]
#     UCBT_mean = (optimal_reward - groups.loc['UCBT'].values).values[0]
#     UCBPI_mean = (optimal_reward - groups.loc['UCBPI'].values).values[0]
#     TS_mean = (optimal_reward - groups.loc['TS'].values).values[0]
#     EG_1_mean = (optimal_reward - groups.loc['EG_1'].values).values[0]
#     EG_2_mean = (optimal_reward - groups.loc['EG_2'].values).values[0]
#     EG_3_mean = (optimal_reward - groups.loc['EG_3'].values).values[0]
    
#     T = np.arange(2000)
#     trial = 10
    
#     plt.figure(figsize=(30,20))
#     sns.set(font_scale=2, style='white')
    
#     plt.plot(EG_1_mean,alpha=1,color='#979DAC', linestyle='--', linewidth=3)
#     plt.plot(EG_2_mean,alpha=1,color='#7D8597', linestyle='--', linewidth=3)
#     plt.plot(EG_3_mean,alpha=1,color='#5C677D', linestyle='--', linewidth=3)
#     plt.plot(TS_mean,alpha=1,color='#33415C', linestyle='--', linewidth=3)
    
#     plt.plot(UCB_1_mean,alpha=1,color='#979DAC', linestyle=':', linewidth=3)
#     plt.plot(UCB_2_mean,alpha=1,color='#7D8597', linestyle=':', linewidth=3)
#     plt.plot(UCBT_mean,alpha=1,color='#5C677D', linestyle=':', linewidth=3)
#     plt.plot(UCBPI_mean,alpha=1,color='#33415C', linestyle=':', linewidth=3)


#     plt.plot(IDS_1_mean,alpha=1,color='#FF0000', linestyle='-', linewidth=4)
#     plt.plot(IDS_2_mean,alpha=1,color='#FF8700', linestyle='-', linewidth=4)
#     plt.plot(IDS_3_mean,alpha=1,color='#FFD300', linestyle='-', linewidth=4)
#     plt.plot(IDST_1_mean,alpha=1,color='#FF8FA3', linestyle='-', linewidth=4)

    
#     if bounds:
#         plt.fill_between(T, get_low(optimal_reward, EG_1), get_high(optimal_reward, EG_1), color='#000000', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, EG_2), get_high(optimal_reward, EG_2), color='#000000', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, EG_3), get_high(optimal_reward, EG_3), color='#000000', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, TS), get_high(optimal_reward, TS), color='#666666', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, UCB_1), get_high(optimal_reward, UCB_1), color='#666600', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, UCB_2), get_high(optimal_reward, UCB_2), color='#666600', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, UCBT), get_high(optimal_reward, UCBT), color='#666600', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, UCBPI), get_high(optimal_reward, UCBPI), color='#666600', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, IDS_1), get_high(optimal_reward, IDS_1), color='#FF6666', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, IDS_2), get_high(optimal_reward, IDS_2), color='#FF6666', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, IDS_3), get_high(optimal_reward, IDS_3), color='#FF6666', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, IDST_1), get_high(optimal_reward, IDST_1), color='#FF6666', alpha=0.1)

        
        
        
#     plt.legend(['EG $ε$=0.05', 'EG $ε$=0.1', 'EG $ε$=0.15', 'TS', 'UCB c=1', 'UCB c=2', 'UCB-tuned',
#                 'UCBPI', 'ACIDP L=2, n=1', 'ACIDP L=4, n=1', 'ACIDP L=2, n=2',
#                 'ACIDP-$θ$ L=2, n=1'],
#               bbox_to_anchor=(0.9, -0.05), ncol=6, fancybox=True)
#     plt.title(title)
#     plt.xlabel('Horizon')
#     plt.ylabel('Cumulative Regret')
#     plt.show()

# def draw_regret(optimal_reward, groups, title, bounds=True):
    
#     IDS_1_mean = (optimal_reward - groups.loc['IDS_1'].values).values[0]
#     IDS_2_mean = (optimal_reward - groups.loc['IDS_2'].values).values[0]
#     IDS_3_mean = (optimal_reward - groups.loc['IDS_3'].values).values[0]
#     IDST_1_mean = (optimal_reward - groups.loc['IDST_1'].values).values[0]
#     IDSN_mean = (optimal_reward - groups.loc['IDSN'].values).values[0]
#     IDSW_mean = (optimal_reward - groups.loc['IDSW'].values).values[0]
#     UCB_1_mean = (optimal_reward - groups.loc['UCB_1'].values).values[0]
#     UCB_2_mean = (optimal_reward - groups.loc['UCB_2'].values).values[0]
#     UCBT_mean = (optimal_reward - groups.loc['UCBT'].values).values[0]
#     UCBPI_mean = (optimal_reward - groups.loc['UCBPI'].values).values[0]
#     TS_mean = (optimal_reward - groups.loc['TS'].values).values[0]
#     EG_1_mean = (optimal_reward - groups.loc['EG_1'].values).values[0]
#     EG_2_mean = (optimal_reward - groups.loc['EG_2'].values).values[0]
#     EG_3_mean = (optimal_reward - groups.loc['EG_3'].values).values[0]
    
#     T = np.arange(2000)
#     trial = 10
    
#     plt.figure(figsize=(30,20))
#     sns.set(font_scale=2, style='white')
    
#     plt.plot(EG_1_mean,alpha=1,color='#979DAC', linestyle='--', linewidth=3)
#     plt.plot(EG_2_mean,alpha=1,color='#7D8597', linestyle='--', linewidth=3)
#     plt.plot(EG_3_mean,alpha=1,color='#5C677D', linestyle='--', linewidth=3)
#     plt.plot(TS_mean,alpha=1,color='#33415C', linestyle='--', linewidth=3)
    
#     plt.plot(UCB_1_mean,alpha=1,color='#979DAC', linestyle=':', linewidth=3)
#     plt.plot(UCB_2_mean,alpha=1,color='#7D8597', linestyle=':', linewidth=3)
#     plt.plot(UCBT_mean,alpha=1,color='#5C677D', linestyle=':', linewidth=3)
#     plt.plot(UCBPI_mean,alpha=1,color='#33415C', linestyle=':', linewidth=3)


# #     plt.plot(IDS_1_mean,alpha=1,color='#FF0000', linestyle='-', linewidth=4)
# #     plt.plot(IDS_2_mean,alpha=1,color='#FF8700', linestyle='-', linewidth=4)
#     plt.plot(IDS_3_mean,alpha=1,color='#FFD300', linestyle='-', linewidth=4)
#     plt.plot(IDST_1_mean,alpha=1,color='#FF8FA3', linestyle='-', linewidth=4)
#     plt.plot(IDSN_mean,alpha=1,color='#147DF5', linestyle='-.', linewidth=4)
#     plt.plot(IDSW_mean,alpha=1,color='#DEFF0A', linestyle='-', linewidth=4)

    
#     if bounds:
#         plt.fill_between(T, get_low(optimal_reward, EG_1), get_high(optimal_reward, EG_1), color='#000000', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, EG_2), get_high(optimal_reward, EG_2), color='#000000', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, EG_3), get_high(optimal_reward, EG_3), color='#000000', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, TS), get_high(optimal_reward, TS), color='#666666', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, UCB_1), get_high(optimal_reward, UCB_1), color='#666600', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, UCB_2), get_high(optimal_reward, UCB_2), color='#666600', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, UCBT), get_high(optimal_reward, UCBT), color='#666600', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, UCBPI), get_high(optimal_reward, UCBPI), color='#666600', alpha=0.05)
#         plt.fill_between(T, get_low(optimal_reward, IDS_1), get_high(optimal_reward, IDS_1), color='#FF6666', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, IDS_2), get_high(optimal_reward, IDS_2), color='#FF6666', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, IDS_3), get_high(optimal_reward, IDS_3), color='#FF6666', alpha=0.1)
#         plt.fill_between(T, get_low(optimal_reward, IDST_1), get_high(optimal_reward, IDST_1), color='#FF6666', alpha=0.1)

        
        
        
#     plt.legend(['EG $ε$=0.05', 'EG $ε$=0.1', 'EG $ε$-0.15', 'TS', 'UCB c=1', 'UCB c=2', 'UCB-tuned',
#                 'UCBPI', 'ACIDP L=2, n=2', 'ACIDP-$θ$ L=2, n=1', 'ACIDP-without Transferability Test L=2, n=1', 'ACIDP-window L=2, n=1'],
#               bbox_to_anchor=(0.98, -0.05), ncol=6, fancybox=True)
#     plt.title(title)
#     plt.xlabel('Horizon')
#     plt.ylabel('Cumulative Regret')
#     plt.show()