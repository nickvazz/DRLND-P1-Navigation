import os
import argparse
import json

import torch
from dqn_agent import Agent
from collections import deque
from unityagents import UnityEnvironment

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('--n_episodes', default=2000)
parser.add_argument('--max_t', default=2000)
parser.add_argument('--eps_start', default=1.0)
parser.add_argument('--eps_end', default=0.01)
parser.add_argument('--eps_decay', default=0.9)
parser.add_argument('--BUFFER_SIZE', default=1e5, type=int)
parser.add_argument('--BATCH_SIZE', default=64, type=int)
parser.add_argument('--GAMMA', default=0.99)
parser.add_argument('--TAU', default=1e-3)
parser.add_argument('--LR', default=5e-4)
parser.add_argument('--UPDATE_EVERY', default=4, type=int)
parser.add_argument('--state_size', default=37, type=int)
parser.add_argument('--action_size', default=4, type=int)
parser.add_argument('--seed', default=42, type=int)
parser.add_argument('--model_num', default=8)
parser.add_argument('--num_units', default=64, type=int)

args = vars(parser.parse_args())
print(args)



for key, value in args.items():
    exec(f'{key} = {value}')

os.system(f'mkdir -p results/model-{model_num}')
with open(f'results/model-{model_num}/training_params.json', 'w') as outfile:
    json.dump(args, outfile)

# # env = UnityEnvironment(file_name="./Banana_Linux/Banana.x86_64")
env = UnityEnvironment(file_name="./Banana_Linux_NoVis/Banana.x86_64", worker_id=int(model_num))
# env = UnityEnvironment(file_name='./Banana_Mac.app', worker_id=int(model_num))


brain_name = env.brain_names[0]
brain = env.brains[brain_name]

agent = Agent(state_size, action_size, seed,
              BUFFER_SIZE, BATCH_SIZE, GAMMA,
              TAU, LR, UPDATE_EVERY, num_units)


def dqn(n_episodes, max_t, eps_start, eps_end, eps_decay):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):

        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]

        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]

            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]

            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break

        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'results/model-{}/checkpoint_{:04d}.pth'.format(model_num, i_episode))

        if np.mean(scores_window)>=200:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'results/model-{}/checkpoint_{:04d}-score-{:.2f}.pth'.format(model_num, np.mean(scores_window) ,i_episode))
            break
    return scores

scores = dqn(n_episodes, max_t, eps_start, eps_end, eps_decay)




# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig('results/model-{}/scores.png'.format(model_num))
# plt.show()

df = pd.DataFrame({'episode':np.arange(len(scores)),'score':scores})
df.set_index('episode', inplace=True)
df.to_csv('results/model-{}/scores.csv'.format(model_num))

os.system('cp model.py results/model-{}/'.format(model_num))

env.close()
