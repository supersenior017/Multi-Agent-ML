import gym
import rware
import time
env = gym.make("rware-tiny-2ag-v1")
observation= env.reset()
print(env.shelfs[0].x, env.goals[0])
env.agents[0].dir

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated = env.step([4,4])
    res = env.render(mode = "rgb_array")
    # print("a", res)
    time.sleep(0.1)
    print('---')
    # if terminated or truncated:
    #     observation= env.reset()
env.close()


# import gym
# import time
# env = gym.make("CartPole-v1")
# observation= env.reset()
# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated = env.step(action)
#     res = env.render(mode="rgb-array")
#     print("a", res)
#     time.sleep(1)
#     # if terminated or truncated:
#     #     observation= env.reset()
# env.close()

# 0 hold
# 1 forward
# 2 anti clockwise
# 3 clock wise
# 4 catch pick