import gymnasium as gym
# 
# env = gym.make("CliffWalking-v0" , render_mode = "human")
# 
# obs , info = env.reset(seed = 0 )
# 
# for i in range(100) :
#     env.step(env.action_space.sample())
#     env.render()



import numpy as np
import matplotlib.pyplot as plt 

from tqdm import tqdm

def QLearning( env ,n , gamma) :
    V = [0]*env.observation_space.n
    ns = [0]*env.observation_space.n
    
    for i in tqdm(range(n)) :
        s = env.reset(seed= 0)[0]

        done = False
        while not done :
            action = env.action_space.sample()
            s_new,r , terminated , truncated , _ = env.step(action)
            V[s] +=  0.1 * (r + gamma*V[s_new] - V[s] )
            ns[s] += 1
            s = s_new
            done = terminated or truncated 
            
    return V , ns 
        
env = gym.make("CliffWalking-v0")

V , ns = QLearning(env , 1000 , 0.9)

nV = np.array(V).reshape(4,12)
nns = np.array(ns).reshape(4,12)
plt.imshow(nV)
plt.colorbar()
plt.show()
