import time

import gym
import panda_gym
from stable_baselines3 import SAC

from gym.wrappers import FlattenObservation, FilterObservation
import torch as th
##
# cuda
##

env = gym.make("PandaPush-v1", render=True)
env = FlattenObservation(FilterObservation(env, ['observation', 'desired_goal']))
observation = env.reset()

done = False

model = SAC("MlpPolicy",
            env,
            verbose=1,
            buffer_size=1_000_000,
            batch_size=256,
            learning_rate=0.003, # learning_rate=0.0003
            learning_starts=1024,
            gamma=0.95,
            ent_coef='auto',
            # policy_kwargs=policy_kwargs,
            train_freq=1024,
            gradient_steps=-1,
            device="cpu")

# zastanowic sie na obrazem
# sprobowac na gpu
# potestowac rozne hyperparametry optuna
# wyprobowac kilka roznych modeli i porownac wyniki
# dodac drugi algorytm HER !
# wiekszy learning rate
# min timesteps=30000
# zmienic wspolczynnik tarcia

timesteps = 30_000
start = time.time()
model.learn(timesteps)
end = time.time()
print("=== LEARN === {}".format(end - start))

total_success = 0
total_episodes = 50
reward = 0

for i_episode in range(1, total_episodes + 1):
    observation = env.reset()
    for t in range(1, 50+1):
        env.render()
        #action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        if done:
            break

    if reward != -1.0:
        total_success += 1
    print(f"Episode={i_episode}", "Success_rate={:.2f}".format((100*total_success)/i_episode), "%")
    reward = 0

env.close()
