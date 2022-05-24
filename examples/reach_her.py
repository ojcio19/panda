import time

import gym
import torch as th
from stable_baselines3.her import HerReplayBuffer
import matplotlib.pyplot as plt
from panda_gym.utils import distance

import panda_gym
from stable_baselines3 import SAC, HER, DQN, DDPG, TD3
from stable_baselines3.sac import MlpPolicy
import pandas
import numpy as np

if th.cuda.is_available():
    print("It works")
else:
    raise Exception("Not working")

model_type = TD3

# DEFINE PARAMETERS
learning_rate = 0.002
gamma = 0.95
arch_values = "[128, 64, 32]"
env = gym.make("PandaReach-v1", render=True)

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[128, 64, 32],
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
print("Starting learning")
timesteps = 2000
# start = time.time()
model.learn(timesteps)
# end = time.time()
# print("=== LEARN === {}".format(end - start))

# SAVE RESULTS
# total_value, counter, result = 0.0, 0, []
# for i in model.ep_success_buffer:
#     total_value += i
#     counter += 1
#     result.append(total_value / counter)
#
# # TRANSFER RESULTS TO CSV
# df = pandas.DataFrame(data={arch_values: result})
# df.to_csv("result2.csv", sep=';', index=False)
#
# # SAVE MODEL
# model.save('reach_her_model')
#
# # LOAD SAVED MODEL
# model = model_type.load('reach_her_model', env=env)

# MAKE PREDICTIONS ON LEARNED MODEL
total_episodes, reward, pred_limit, total_success, sum_time = 20, 0, 30, 0, 0

x_ball, y_ball, z_ball = [], [], []
x_ball_new, y_ball_new, z_ball_new = [], [], []
diff_y = []

for i_episode in range(1, total_episodes + 1):
    observation = env.reset()
    for t in range(1, pred_limit + 1):
        # SHOW CAMERA IMAGE

        # start = time.time()
        point_y, point_z, robot_y, robot_z = env.render(mode="point_side")
        # end = time.time()
        _, point_x, _, robot_x = env.render(mode="point_front")
        # end2 = time.time()
        # diff_y.append(np.abs(point_y_side-point_y_front))
        # point_y = float(((point_y_side + point_y_front) / 2))
        # robot_x = float(((robot_x_side + robot_x_front) / 2))

        ball = np.array([point_x, point_y, point_z])
        robot = np.array([robot_x, robot_y, robot_z])
        print(distance(ball, robot))
        # img = env.render(mode="front")
        # plt.imshow(img)
        # print("robot", robot)
        # plt.show()
        if distance(ball, robot) < 0.05:
            img = env.render(mode="front")
            plt.imshow(img)
            print("triggered!", distance(ball, robot))
            print("Coordinates!", 'b:', ball, 'r:', robot)
            #print("ball:", ball)
            #print("robot", robot)
            plt.show()
        # print("Calc:", ls)
        # SHOW COORDINATES OF TARGET
        # print("x:", point_z+75, "y:", point_x, "z:", point_y, "time side:", end2-start)

        # print("time side:", end2 - start)
        # sum_time = sum_time + end2 - start

        # PREDICT A MOVE
        x_ball.append(ball[0])
        y_ball.append(ball[1])
        z_ball.append(ball[2])

        x_ball_new.append(observation['desired_goal'][0])
        y_ball_new.append(observation['desired_goal'][1])
        z_ball_new.append(observation['desired_goal'][2])

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

print("Diff y ball:", np.mean(diff_y), "Maks", max(diff_y))

print("Max initial", 'x', max(x_ball), 'y', max(y_ball), 'z', max(z_ball))
print("Min initial", 'x', min(x_ball), 'y', min(y_ball), 'z', min(z_ball))

print("Max new", 'x', max(x_ball_new), 'y', max(y_ball_new), 'z', max(z_ball_new))
print("Min new", 'x', min(x_ball_new), 'y', min(y_ball_new), 'z', min(z_ball_new))

print("RES TIME", sum_time / total_episodes, "in total:", sum_time)
env.close()
