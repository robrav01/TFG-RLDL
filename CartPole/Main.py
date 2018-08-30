import gym
from CartPole.DQN import DQN

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.95
exploration = 1.0

# Training parameters
EPISODES = 50000
STEPS = 1000

if __name__ == '__main__':
    # The problem to solve
    env = gym.make('CartPole-v1')

    # The agent that solves it
    DQN = DQN(env.observation_space.shape[0],   # State's dimensions
              env.action_space.n,               # Number of available actions
              learning_rate,
              discount_factor,
              exploration)

    for e in range(EPISODES):
        state = env.reset()
        total_reward = 0

        for t in range(STEPS):
            action = DQN.act(state)

            if e > 500:
                env.render()

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

        DQN.replay(32)  # Experience replay with a batch of 32 samples
