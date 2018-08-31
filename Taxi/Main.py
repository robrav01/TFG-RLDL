import gym
from Taxi.DQN import DQN
import time
import os

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.95
exploration = 1.0

# Training parameters
EPISODES = 4000

if __name__ == '__main__':
    # The problem to solve
    env = gym.make('Taxi-v2')

    # The agent that solves it
    DQN = DQN(500,   # State's dimensions
              env.action_space.n,               # Number of available actions
              learning_rate,
              discount_factor,
              exploration)

    reward_list = []

    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0

        done = False
        DQN.memory.clear()

        while not done:
            action = DQN.act(state)

            # Uncomment lines below to render the GUI
            '''
            os.system('clear')
            time.sleep(0.2)
            env.render()
            '''
            next_state, reward, done, _ = env.step(action)

            DQN.add_to_memory((state, action, next_state, reward, done))

            total_reward += reward

            if done:
                print('Episode: {}'.format(e),
                      'Total reward: {}'.format(total_reward),
                      'Explore P: {:.4f}'.format(DQN.epsilon))
                state = env.reset()
                total_reward = 0
            else:
                state = next_state

        DQN.replay()  # Experience replay with a batch of 32 samples