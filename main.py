# ============================================================
# Traffic Signal Control using Deep Q-Network (DQN)
# Includes:
# - Fixed Baseline Comparison
# - Training Loop
# - Model Saving
# - Evaluation Mode
# - Reward Plotting
# ============================================================

from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent
from memory.replay_buffer import ReplayBuffer

import numpy as np
import matplotlib.pyplot as plt
import torch


# ============================================================
# FIXED-TIMING BASELINE CONTROLLER
# ============================================================

def run_fixed_baseline(env, steps=200):
    """
    Runs a simple fixed phase switching controller.
    Used as a benchmark for comparison.
    """
    state = env.reset()
    total_waiting = 0

    phase = 0
    phase_duration = 20
    counter = 0

    for _ in range(steps):

        if counter >= phase_duration:
            phase = (phase + 1) % env.num_phases
            counter = 0

        _, reward, done = env.step(phase)

        # Reverse delta reward to approximate waiting accumulation
        total_waiting += -reward
        counter += 1

    return total_waiting


# ============================================================
# DQN EVALUATION MODE (No Learning, No Exploration)
# ============================================================

def run_dqn_evaluation(env, agent, steps=200):
    """
    Runs trained DQN with epsilon = 0
    to measure true learned performance.
    """
    state = env.reset()
    total_waiting = 0
    done = False

    # Disable exploration
    agent.epsilon = 0.0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)

        total_waiting += -reward
        state = next_state

    return total_waiting


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

if __name__ == "__main__":

    # ---------------------------
    # Initialize Environment
    # ---------------------------
    env = TrafficEnv()
    env.start()

    state_dim = len(env.get_state())
    action_dim = env.num_phases

    agent = DQNAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(50000)

    episodes = 500
    episode_rewards = []

    # ============================================================
    # BASELINE COMPARISON
    # ============================================================
    print("Running fixed baseline...")
    baseline_wait = run_fixed_baseline(env)
    print("Baseline total waiting:", round(baseline_wait, 2))
    print("-" * 50)

    # ============================================================
    # DQN TRAINING LOOP
    # ============================================================

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

        episode_rewards.append(total_reward)

        print(
            f"Episode {episode+1}, "
            f"Total Reward: {round(total_reward, 2)}, "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    print("Training completed.")

    # ============================================================
    # SAVE TRAINED MODEL
    # ============================================================
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("Model saved as dqn_model.pth")

    # ============================================================
    # EVALUATION MODE
    # ============================================================
    print("-" * 50)
    print("Running DQN evaluation...")

    dqn_wait = run_dqn_evaluation(env, agent)

    print("Baseline total waiting:", round(baseline_wait, 2))
    print("DQN total waiting:", round(dqn_wait, 2))
    print("Average reward (last 50 episodes):",
          round(np.mean(episode_rewards[-50:]), 3))

    # ============================================================
    # TRAINING PERFORMANCE PLOT
    # ============================================================

    plt.figure(figsize=(10, 5))

    # Raw episode rewards
    plt.plot(episode_rewards, label="Raw Reward")

    # Moving average smoothing
    window = 10
    if len(episode_rewards) >= window:
        moving_avg = np.convolve(
            episode_rewards,
            np.ones(window) / window,
            mode='valid'
        )

        plt.plot(
            range(window - 1, len(episode_rewards)),
            moving_avg,
            label="Moving Avg (10)"
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("DQN Training Performance")
    plt.legend()
    plt.show()

    env.close()