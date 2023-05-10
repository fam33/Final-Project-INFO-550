import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras import backend as K
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

#Class for the Deep Q Learning Agent.
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01 #minimum exploration
        self.epsilon_decay = 0.995 
        self.learning_rate = 0.001
        self.model = self._build_model()

    #Function to build the model using neural network for Deep Q.
    def _build_model(self):
        #Sequential Neural network for Deep-Q learning Model,
        model = Sequential()
        #Setting dimensions to the state size of the environment.
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        #Fully connected layer with 24 neurons and ReLU activation function to the model
        model.add(Dense(24, activation='relu'))
        #Linear activation function to predict Q values.
        model.add(Dense(self.action_size, activation='linear'))
        #Compiling the model with mean squared error as the loss function and Adam optimizer with specified learning rate.
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        #This will append all the actions, states, etc in to memory.
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #With a probability of self.epsilon, choosing a random action.
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        #Getting the predicted Q-values for each action.
        act_values = self.model.predict(state)
        #Choosing the action with the highest predicted Q-value
        return np.argmax(act_values[0]) 

    def replay(self, batch_size):
        #Random sample of memory for batch_size.
        minibatch = random.sample(self.memory, batch_size)
        #Iterating over the minibatch memory
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                #Calculating the target value of current state using Bellman equation.
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            #Using the Q value for the chosen action.
            target_f[0][action] = target
            #Training the model for the given state and Q value
            self.model.fit(state, target_f, epochs=1, verbose=0)
            #Checking if it's greater than the minimum value of epsilon.
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
