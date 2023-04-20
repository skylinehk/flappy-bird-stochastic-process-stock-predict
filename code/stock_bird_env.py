import gym
from gym import spaces
import numpy as np
import matplotlib.pyplot as plt

class StockBirdEnv(gym.Env):
    def __init__(self):
        # Define observation space (single continuous variable)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)

        # Define action space (discrete with 3 actions)
        self.action_space = spaces.Discrete(3)

        # Set initial reward and step count
        self.reward = 100
        self.steps = 0

    def reset(self):
        # Reset the environment to its initial state
        self.reward = 100
        self.steps = 0
        self.trajectory = np.random.normal(loc=0, scale=1, size=(1000,))

        # Return the initial state as an observation
        return np.array([0])

    def step(self, action):
        # Update the state of the environment based on the agent's action
        self.steps += 1

        # Get the current price tick of the stock
        price_tick = np.random.normal(loc=0, scale=1)

        # Calculate the distance from the ideal trajectory
        distance = np.abs(price_tick - self.trajectory[self.steps])

        # Calculate the reward based on the distance from the trajectory
        if price_tick > self.trajectory[self.steps]:
            reward = distance
        else:
            reward = -distance

        # Update the cumulative reward
        self.reward += reward

        # Check if the episode is over
        done = self.reward < 0 or self.steps >= 1000

        # Return the new state, reward, and done flag
        return np.array([price_tick]), reward, done, {}

    def render(self, mode='human'):
        # Plot the ideal trajectory and the bird's position
        plt.plot(self.trajectory[:self.steps+1], color='gray')
        plt.plot(self.steps, self.observation[0], 'ro', markersize=10)

        # Add labels and formatting
        plt.title('Stock Bird Environment')
        plt.xlabel('Time Step')
        plt.ylabel('Price Tick')
        plt.ylim([-5, 5])
        plt.xlim([0, 1000])

        # Show the plot
        plt.show()


