import gym
import numpy as np

from dqn import DQN
from progress_viewer import ProgressViewer

env = gym.make("CartPole-v1")
gamma = 0.9
epsilon = .95

trials = 1000000
trial_len = 500

updateTargetNetwork = 2000
dqn_agent = DQN(env=env)
viewer = ProgressViewer(True)

update_step_counter = 0
for trial in range(trials):
    cur_state = env.reset().reshape(1, 4)
    total_episode_reward = 0

    for step in range(trial_len):
        action = dqn_agent.act(cur_state)

        new_state, reward, done, _ = env.step(action)
        total_episode_reward += reward

        reward = reward if not done else -20

        new_state = new_state.reshape(1, 4)
        dqn_agent.remember(cur_state, action, reward, new_state, done)

        dqn_agent.replay()  # internally iterates default (prediction) model
        #         dqn_agent.target_train() # iterates target model
        update_step_counter += 1
        if update_step_counter == updateTargetNetwork:
            dqn_agent.target_train()
            update_step_counter = 0

        cur_state = new_state
        if done:
            viewer.add(step, dqn_agent.epsilon, dqn_agent.history)
            break
    if trial % 10 == 0:
        print('trial:{}, mean_reward:{}'.format(trial, np.mean(viewer.rewards[-10:])))