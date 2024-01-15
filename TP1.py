import bandits
import numpy as np
import random
import matplotlib.pyplot as plt
import math 
# initialisation du générateur de nombres pseudo-aléatoires
random.seed(0)
np.random.seed(0)


def random_strategy(bandit, n):
    cumulative_regret = 0
    regrets = []
    for i in range(n):
        chosen_arm = random.randint(0, bandit.nbr_arms - 1)
        pk = bandit.arms[chosen_arm].mean
        p_star = bandit.best_reward
        regret = p_star - pk
        cumulative_regret += regret
        regrets.append(cumulative_regret)
    return regrets 
    


def graph(lists , label):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

    for i, lst in enumerate(lists, 1):
        plt.plot(range(1, len(lst) + 1), lst)

    plt.xlabel('X Axis Label')
    plt.ylabel('Y Axis Label')
    plt.title(label)
    plt.legend()  # Show legend for different lists

    plt.show()

n = 100
b = bandits.BernoulliBandit(np.array([0.3, 0.42, 0.4]) , seed = 42)
bi_regrets_random = []
for i in range(n) :
    bi_regrets_random.append(random_strategy(b, 1000))





def follow_the_leader(bandit , n , k ) :
    bras_returns = []
    for i in range(len(bandit.arms)) :
        bras_returns.append([])
    cumulative_regret = 0
    regrets = []
    for _ in range(k) :
        for i in range(len(bandit.arms)) :
            reward = bandit.pull(i)
            bras_returns[i].append(reward)
            pk = bandit.arms[i].mean
            p_star = bandit.best_reward
            regret = p_star - pk
            cumulative_regret += regret
            regrets.append(cumulative_regret)
    
    np_bras_returns = np.array(bras_returns)
    mus = np.mean(np_bras_returns , axis=1)
    chosen_arm = np.argmax(mus)
    
    long =  n - k*len(bandit.arms)
    for i in range(long):
        pk = bandit.arms[chosen_arm].mean
        p_star = bandit.best_reward
        regret = p_star - pk
        cumulative_regret += regret
        regrets.append(cumulative_regret)
    return regrets 
 
b = bandits.BernoulliBandit(np.array([0.3, 0.42, 0.4]) , seed = 42)
bi_regrets_follow = []



for i in range(n) :
    bi_regrets_follow.append(follow_the_leader(b, 1000 , 100))


def compare(l ) :
    means = []
    for strat in l :
        
        np_strat = np.array(strat)
        strat_means = np.mean(np_strat, axis=0 )
        means.append(strat_means.tolist())
        
    return means



def eps_glouton(bandit , n , eps , dec_rate) :
    si = np.zeros(len(bandit.arms))
    ni = np.zeros(len(bandit.arms))
    mask = ni != 0
    cumulative_regret = 0
    regrets = []
    for i in range(n) :
        r = np.random.rand(1)[0]
        if r < eps :
            chosen_arm = random.randint(0, bandit.nbr_arms - 1)
        else :
            
            chosen_arm = np.argmax(si/ni)
        
        
        ni[chosen_arm] = ni[chosen_arm] + 1
        
        reward = bandit.pull(chosen_arm)
        si[chosen_arm] = si[chosen_arm] + reward

        pk = bandit.arms[chosen_arm].mean
        p_star = bandit.best_reward
        regret = p_star - pk
        cumulative_regret += regret
        regrets.append(cumulative_regret)
        
        eps = eps *dec_rate
    return regrets


b = bandits.BernoulliBandit(np.array([0.3, 0.42, 0.4]) , seed = 42)
bi_regrets_eps = []

for i in range(n) :
    bi_regrets_eps.append(eps_glouton(b, 1000 , 1 , 0.99 ))
    
    

def proportional_strategy(bandit, n):
    
    chosen = [0] * bandit.nbr_arms  # Initial sum of rewards for each arm
    rewards = [0]  * bandit.nbr_arms
    regrets = []
    cumulative_regret = 0
    for i in range(n):
        if 0 in chosen :
            chosen_arm = random.choice(range(bandit.nbr_arms))
            
        else :
            probabilities = [rewards[b]/ chosen[b] for b in range(len(chosen))]
            chosen_arm = np.argmax(probabilities)
        chosen[chosen_arm] += 1
        reward = bandit.pull(chosen_arm)
        rewards[chosen_arm] += reward
        pk = bandit.arms[chosen_arm].mean
        p_star = bandit.best_reward
        regret = p_star - pk
        cumulative_regret += regret
        regrets.append(cumulative_regret)

    return regrets

b = bandits.BernoulliBandit(np.array([0.3, 0.42, 0.4]) , seed = 42)
bi_regrets_prop = []

for i in range(n) :
    bi_regrets_prop.append(proportional_strategy(b, 1000 ))
    

def boltzman_strategy(bandit, n , tau):
    
    chosen = [0] * bandit.nbr_arms  # Initial sum of rewards for each arm
    rewards = [0]  * bandit.nbr_arms
    regrets = []
    cumulative_regret = 0
    for i in range(n):
        if 0 in chosen :
            probs = [1 / bandit.nbr_arms] * bandit.nbr_arms 
        else :
            probs = np.array(rewards)/np.array(chosen)
            
        qs = np.exp(np.array(probs) / tau)
        probabilities = np.array(qs)/sum(qs)
        chosen_arm = random.choices(range(bandit.nbr_arms), weights=probabilities)[0]
        chosen[chosen_arm] += 1
        reward = bandit.pull(chosen_arm)
        rewards[chosen_arm] += reward
        pk = bandit.arms[chosen_arm].mean
        p_star = bandit.best_reward
        regret = p_star - pk
        cumulative_regret += regret
        regrets.append(cumulative_regret)

    return regrets





b = bandits.BernoulliBandit(np.array([0.3, 0.42, 0.4]) , seed = 42)
bi_regrets_boltz = []

for i in range(n) :
    bi_regrets_boltz.append(boltzman_strategy(b, 1000  , 0.1))


def UCB(bandit, n , alpha):
    
    chosen = [0] * bandit.nbr_arms  # Initial sum of rewards for each arm
    rewards = [0]  * bandit.nbr_arms
    regrets = []
    cumulative_regret = 0
    
    for i in range(n):

        probs = np.array(rewards)/ np.array(chosen)         
        
        
        probabilities = np.array(probs) + np.sqrt((alpha*(i+1)/np.array(chosen)))
        chosen_arm = random.choices(range(bandit.nbr_arms), weights=probabilities)[0]
        chosen[chosen_arm] += 1
        reward = bandit.pull(chosen_arm)
        rewards[chosen_arm] += reward
        pk = bandit.arms[chosen_arm].mean
        p_star = bandit.best_reward
        regret = p_star - pk
        cumulative_regret += regret
        regrets.append(cumulative_regret)

    return regrets



b = bandits.BernoulliBandit(np.array([0.3, 0.42, 0.4]) , seed = 42)
bi_regrets_UCB = []

for i in range(n) :
    bi_regrets_UCB.append(UCB(b, 1000  , 0.1))





to_cmp = [bi_regrets_follow , bi_regrets_random , bi_regrets_eps , bi_regrets_prop , bi_regrets_boltz , bi_regrets_UCB]


names = ["follow the lead" , "random" , "eps-glouton" , "proprtional" , "boltzman"  , "UCB" ]

cmp = compare(to_cmp)


for i in range(len(to_cmp)) :
    plt.plot(cmp[i] , label = names[i])
 
# print("starting" )
# l = []
# names = []
# for i in range(1, 100) : 
#     b = bandits.BernoulliBandit(np.array([0.3, 0.42, 0.4]) , seed = 42)
#     bi_regrets_eps = []
# 
#     for _ in range(n) :
#         bi_regrets_eps.append(eps_glouton(b, 1000 , 1 , i/100 ))
#     l.append(bi_regrets_eps)
#     names.append(str(i))
#     print(i/100) 
# 
# cmp = compare(l)



# enumerated_lists = list(enumerate(cmp))
# 
# # Sort the enumerated list based on the last element of each list
# sorted_enumerated_lists = sorted(enumerated_lists, key=lambda x: x[1][-1])
# 
# best_10 = sorted_enumerated_lists[:10]
# 
# print("plotting")
# for i in range(len(best_10)) :
#     plt.plot(best_10[i][1] , label = str(best_10[i][0]/100))
    

plt.xlabel("Picks")
plt.ylabel("Regret")
plt.title("Comparison of strats")
plt.legend()
plt.show()
