import time

import gym
import torch as th
from stable_baselines3.her import HerReplayBuffer
import matplotlib.pyplot as plt

import panda_gym
from stable_baselines3 import SAC, HER, DQN, DDPG, TD3
from stable_baselines3.sac import MlpPolicy
import pandas

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
timesteps = 1000
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

# TRANSFER RESULTS TO CSV
df = pandas.DataFrame(data={arch_values: result})
df.to_csv("result2.csv", sep=';', index=False)

# SAVE MODEL
model.save('reach_her_model')

# LOAD SAVED MODEL
model = model_type.load('reach_her_model', env=env)

# MAKE PREDICTIONS ON LEARNED MODEL
total_episodes, reward, pred_limit = 4, 0, 1
total_success = 0

y_lowest, x_lowest = 100, 100
y_highest, x_highest = -100, -100
sum_time = 0
for i_episode in range(1, total_episodes + 1):
    observation = env.reset()

    for t in range(1, pred_limit + 1):
        # SHOW CAMERA IMAGE
        img = env.render(mode="front")
        plt.imshow(img)

        start = time.time()
        point_x_side, point_y = env.render(mode="point_side")
        end = time.time()
        point_x_front, point_z = env.render(mode="point_front")
        end2 = time.time()
        point_x = float(((point_x_side + point_x_front) / 2) + 35)
        # SHOW COORDINATES OF TARGET
        # print("x:", point_z+75, "y:", point_x, "z:", point_y, "time side:", end2-start)

        print("time side:", end2 - start)
        sum_time = sum_time + end2 - start
        # plt.show()

        # PREDICT A MOVE
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        if reward > -5:
            total_success += 1

        if done:
            break

    print(f"Episode={i_episode}", "Success_rate = {:.2f}%".format(100 * total_success / total_episodes))
    reward = 0
    total_success = 0

print("RES TIME", sum_time / total_episodes, "in total:", sum_time)
env.close()
