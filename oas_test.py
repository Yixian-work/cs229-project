from cs229_project_scenario import ObstacleAvoidanceScenario
import numpy as np
import gym
import time
from matplotlib import pyplot as plt

if __name__ == "__main__":
    oas = ObstacleAvoidanceScenario()
    #oas = gym.make()
    dt = 0.1
    u = (0., 0.)
    total_reward = 0.
    plot_reward = []
    for k in range(10):
        total_reward = 0.
        oas.reset()
        while True:
            state, reward, if_reset, non_defined = oas.step(u) # move one time step and get the tuple of data
            oas.render() # Test case
            # while time.time() - t < env.dt:
            #     pass
            # We should apply the reinforcement method here!
            # For example, policy = ReinforcementLearning() # (angular velocity, linear velocity)
            #              oas.ego.set_control(policy)
            total_reward += oas._get_reward()
            if if_reset:
                oas.close()
                print("The total reward for this episode is: ", total_reward)
                plot_reward.append(total_reward)
                break
            time.sleep(dt)
    plt.plot(plot_reward)
    plt.savefig('Reward_demo')