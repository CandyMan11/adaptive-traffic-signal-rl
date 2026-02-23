from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent
from memory.replay_buffer import ReplayBuffer

import numpy as np


if __name__ == "__main__":

    env = TrafficEnv()
    env.start()

    # Since state = [vehicle_count, avg_wait]
    state_dim = 2

    # We defined 4 traffic light phases
    action_dim = 4

    agent = DQNAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(10000)

    episodes = 200

    for episode in range(episodes):

        state = env.reset()
        total_reward = 0
        done = False

        while not done:

            action = agent.select_action(state)
            next_state, reward, done = env.step(action)

            replay_buffer.store(state, action, reward, next_state, done)
            agent.train(replay_buffer, batch_size=64)

            state = next_state
            total_reward += reward

        agent.update_target()

        print(
            f"Episode {episode+1}, "
            f"Total Reward: {round(total_reward, 2)}, "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    env.close()
    print("Training completed.")