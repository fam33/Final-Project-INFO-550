import numpy as np
import cv2
import gym
import random
import time
from collections import deque
from gym import spaces

#This is the length to the goal of snake and we assume it to be 30.
snakelg = 30

def encountering_with_food(food_pos, score):
    #Once the food is encountered it's shifted to a random position.
    food_pos = [random.randrange(1,50)*10,random.randrange(1,50)*10]
    #And then the score is implemented by one which is very different from the reward in reinforcement learning.
    score += 1
    return food_pos, score

def encountering_with_boundaries(snake_h):
    #Checking if the snake has gone out of boundaries assigned.
    if snake_h[0]>=500 or snake_h[0]<0 or snake_h[1]>=500 or snake_h[1]<0 :
        return 1
    else:
        return 0

def encountering_with_self(snake_pos):
    #Initializing snake head to the body position.
    snake_h = snake_pos[0]
    #Checking if the snake head has collided with the body.
    if snake_h in snake_pos[1:]:
        return 1
    else:
        return 0

#Defining the environment class with built in gym environment functions.
class SnakeEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    #Function to initialize spaces in the environement.
    def __init__(self):
        super(SnakeEnv, self).__init__()
        # Defineing action space and observation space as they both have to be gym attributes in terms of spaces and discrete for actions.
        self.action_space = spaces.Discrete(4)
        # Spaces box function with high and low values.
        self.observation_space = spaces.Box(low=-500, high=500, shape=(5+snakelg,), dtype=np.float32)

    #Function to specify what happens at each step in the game. 
    def step(self, action):
        #We are appending previous actions to the actions list.
        self.prev_actions.append(action)
        cv2.imshow('Game',self.img)
        cv2.waitKey(1)
        #Setting dtype as unint8 with limit from 0 to 255 for each pixel.
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Displaying food on the grid.
        points = np.array([(self.food_pos[0]+5, self.food_pos[1]), 
                           (self.food_pos[0], self.food_pos[1]+10), 
                           (self.food_pos[0]+10, self.food_pos[1]+10)])
        cv2.drawContours(self.img, [points], 0, (255,192,203), -1)
        # Displaying Snake on the grid.
        for pos in self.snake_pos:
            cv2.circle(self.img, (pos[0] + 5, pos[1] + 5), 5, (255, 255, 0), thickness=3)

        #Creating a delay of 0.05 seconds.
        time.sleep(0.05)

        #Mapping action to position the snake head along with actions.
        ars = {0: [-10, 0], 1: [10, 0], 2: [0, -10], 3: [0, 10]}
        # Changing the head position based on the direction after resetting.
        ar = ars[action]
        self.snake_h[0] += ar[0]
        self.snake_h[1] += ar[1]
        
        food_reward = 0
        # Increasing the Snake length on eating food by checking if the positions of snake and the food are same.
        if self.snake_h == self.food_pos:
            self.food_pos, self.score = encountering_with_food(self.food_pos, self.score)
            self.snake_pos.insert(0,list(self.snake_h))
            food_reward = 10000
        else:
            #If not encountered by food, keeping the lenght of snake same with no increament.
            self.snake_pos.insert(0,list(self.snake_h))
            #Popping the tail after each moment.
            self.snake_pos.pop()

        # On encountering a boundary kill the snake and print the score.
        if encountering_with_boundaries(self.snake_h) == 1 or encountering_with_self(self.snake_pos) == 1:
            font = cv2.FONT_ITALIC
            self.img = np.zeros((500,500,3),dtype='uint8')
            #Pringting a text message with your score of the game.
            cv2.putText(self.img,'Your Score is {}'.format(self.score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
            cv2.imshow('Game',self.img)
            self.done = True
            
        #Getting the eucledian distance from the snake to food to increase the reward for being near to food.
        euclidean_dist_to_food = np.linalg.norm(np.array(self.snake_h) - np.array(self.food_pos))
        #To scale down the reward we divide it by 10 after initializing it to be 250.
        self.total_reward = ((250 - euclidean_dist_to_food) + food_reward)/100
        #Can also print these reward if necessary at each step.
        #print(self.total_reward)
        #Only way to get the reward is to percieve food.
        self.reward = self.total_reward - self.prev_reward
        self.prev_reward = self.total_reward
        #Setting a condition for the reward to make the model learn better.
        if self.done:
            self.reward = -10
        info = {}
        #Initializing arrays for head of snake coordinates.
        h_x = self.snake_h[0]
        h_y = self.snake_h[1]
        #Initializing food coordinates with arrays.
        food_ar_x = h_x - self.food_pos[0]
        food_ar_y = h_y - self.food_pos[1]
        snake_len = len(self.snake_pos)
        #observations with coordinates of heads, food with length and previous actions.
        observation = [h_x, h_y, food_ar_x, food_ar_y, snake_len] + list(self.prev_actions)
        observation = np.array(observation)
        return observation, self.total_reward, self.done, info
    
    #Built in gym function to reset the game after it hits the wall.
    def reset(self):
        #Initializing a boolean variable done to notify if the game is over.
        self.done = False
        self.img = np.zeros((500,500,3),dtype='uint8')
        # Starting Snake and food positions.
        self.snake_pos = [[250,250],[240,250],[230,250]]
        self.food_pos = [random.randrange(1,50)*10,random.randrange(1,50)*10]
        #The game starts with a 0 score.
        self.score = 0
        self.prev_button_direction = 1
        self.button_direction = 1
        #Defining the limit because we have chosen unsigned integer 8 type which has a limit of 255.
        self.snake_h = [250,250]
        #We start the game with a reward of 0 since no actions have been made.
        self.reward = 0
        self.prev_reward = 0
        
        #Initializing head of the snake and food postions after resetting.
        h_x = self.snake_h[0]
        h_y = self.snake_h[1]
        food_ar_x = h_x - self.food_pos[0]
        food_ar_y = h_y - self.food_pos[1]
        #Making sure snake doesn't directly land on the food after resetting.
        snake_len = len(self.snake_pos)
        self.prev_actions = deque(maxlen = snakelg)
        for i in range(snakelg):
            self.prev_actions.append(-1)
        #Observations for storing head coordinates, food coordinates and all previous actions.  
        observation = [h_x, h_y, food_ar_x, food_ar_y, snake_len] + list(self.prev_actions)
        #Passing it into a numpy array.
        observation = np.array(observation)
        return observation  
