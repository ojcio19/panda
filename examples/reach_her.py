import time

import gym
import torch as th
from stable_baselines3.her import HerReplayBuffer
import matplotlib.pyplot as plt
from panda_gym.utils import distance

from stable_baselines3 import SAC, HER, DQN, DDPG, TD3
from stable_baselines3.sac import MlpPolicy
import pandas
import numpy as np
import os

# net_archs = [[64, 64],
#              [128, 128],
#              [256, 256],
#              [64, 64, 64],
#              [128, 128, 128],
#              [256, 256, 256],
#              [128, 64, 32]]

#model_types = [DDPG, SAC, TD3]
model_types = [SAC]
net_arch = [128, 64, 32]
#for net_arch in net_archs:

for model_type in model_types:
    # DEFINE PARAMETERS
    learning_rate = 0.002
    gamma = 0.95
    arch_values = str(net_arch)
    env = gym.make("PandaReach-v1", render=True)

    name = str(model_type)
    name_of_model = name.split('.')[2]

    policy_kwargs = dict(
        activation_fn=th.nn.ReLU,
        net_arch=net_arch,
    )

    observation = env.reset()
    done = False

    # CREATE MODEL
    model = model_type(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            max_episode_length=100,
            n_sampled_goal=4,
            goal_selection_strategy='future',
        ),
        verbose=1,
        buffer_size=100000,
        batch_size=256,
        learning_rate=learning_rate,
        learning_starts=1000,
        gamma=gamma,
        policy_kwargs=policy_kwargs
    )

    # LEARNING
    print("Starting learning, model =", name_of_model)
    timesteps = 5000
    start = time.time()
    model.learn(timesteps)
    end = time.time()
    print("=== LEARN === {}".format(end - start))

    # SAVE RESULTS
    total_value, counter, result = 0.0, 0, []

    for i in model.ep_success_buffer:
        total_value += i
        counter += 1
        result.append(total_value / counter)
        print("Counter:", counter, "Success buffer latest value", i, "Average:", result[-1])

    time_nam = name_of_model + "_time"
    # TRANSFER RESULTS TO CSV
    if os.path.exists("result.csv"):
        df = pandas.read_csv("result.csv", sep=';')
        df[name_of_model] = result
        df[time_nam] = model.time_measured
        df.to_csv("result.csv", sep=';', index=False)
    else:
        df = pandas.DataFrame(data={name_of_model: result})
        df[time_nam] = model.time_measured
        df.to_csv("result.csv", sep=';', index=False)
    #
    # # SAVE MODEL
    # model.save('reach_her_model')
    #
    # # LOAD SAVED MODEL
    # model = model_type.load('reach_her_model', env=env)

    # MAKE PREDICTIONS ON LEARNED MODEL
    total_episodes, reward, pred_limit, total_success, sum_time = 3, 0, 20, 0, 0

    for i_episode in range(1, total_episodes + 1):
        observation = env.reset()
        for t in range(1, pred_limit + 1):
            # SHOW CAMERA IMAGE IF DISTANCE < 2
            point_y, point_z, robot_y, robot_z = env.render(mode="point_side")
            _, point_x, _, robot_x = env.render(mode="point_front")

            # COORDINATES
            ball = np.array([point_z, point_x, point_y])
            robot = np.array([robot_z, robot_x, robot_y])
            
            #if distance(ball, robot) < 2:
            img = env.render(mode="front")
            plt.imshow(img)
            plt.show()

            img = env.render(mode="side")
            plt.imshow(img)
            plt.show()
            
            # PREDICT A MOVE
            action, _states = model.predict(observation)
            observation, reward, done, info = env.step(action)
            # print("Obse:", observation['observation'][3:])
            if reward > -5:
                total_success += 1

            if done:
                break
        print()
        print(f"Episode={i_episode}", "Success_rate = {:.2f}%".format(100 * total_success / total_episodes))
        reward = 0
        total_success = 0

    env.close()

