#Importing Snake Environment.
from SnakeEnv import SnakeEnv
#Setting environment and episodes for testing.
env = SnakeEnv()
episodes = 30
#Looping through each episode.
for episode in range(episodes):
    #Resetting the environment for each episode.
    done = False
    obs = env.reset()
    #Looping through each step of the episode.
    while not done:
        #Taking a random action from the available action space.
        rand_action = env.action_space.sample()
        print("action",rand_action)
        #Taking the action and receiving the next observation, reward, and done status.
        obs, reward, done, info = env.step(rand_action)
        print('reward',reward)
