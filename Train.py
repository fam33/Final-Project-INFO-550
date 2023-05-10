import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import random
import torch
import gym
from gym import spaces
from collections import deque
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from SnakeEnv import SnakeEnv
from DQNAgent import DQNAgent
env = SnakeEnv()

#Class Trainer for training the DQAgent based on the SnakeEnv.
class Trainer:
    def __init__(self, env, agent, episodes=10, batch_size=32):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.batch_size = batch_size
        self.scores = []

    #Function to train.
    def train(self):
        for e in range(self.episodes):
            #Resetting the environment and getting the initial state.
            state = self.env.reset()
            #Reshaping the state for good compatibility with the neural network.
            state = np.reshape(state, [1, self.agent.state_size])
            done = False
            time = 0
            #Loop until the episode is done.
            while not done:
                #Getting the action for current state.
                action = self.agent.act(state)
                next_state, reward, done, _ = self.env.step(action)
                #Reshaping the next state for compatibility.
                next_state = np.reshape(next_state, [1, self.agent.state_size])
                #Remember the current state, action, reward, next state, and whether or not the episode is done.
                self.agent.remember(state, action, reward, next_state, done)
                state = next_state
                time += 1
                #Check if the episode is done.
                if done:
                    #Adding the scores to the scores list.
                    self.scores.append(time)
                    print("episode: {}/{}, score: {}, e: {:.2}".format(e, self.episodes, time, self.agent.epsilon))
            #Check if the memory contains more than the batch size.
            if len(self.agent.memory) > self.batch_size:
                #Using the replay function to update the neural network based on past experiences
                self.agent.replay(self.batch_size)
        #Plotting the scores vs episodes graph.
        plt.plot(self.scores)
        plt.title('Performance plot')
        plt.xlabel('Episodes')
        plt.ylabel('Score')
        plt.show()


if __name__ == "__main__":
    env = SnakeEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    trainer = Trainer(env, agent)
    trainer.train()