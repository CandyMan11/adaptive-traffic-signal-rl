# ============================================================
# Adaptive Traffic Signal Control using DQN
# ============================================================

from env.traffic_env import TrafficEnv
from agents.dqn_agent import DQNAgent
from memory.replay_buffer import ReplayBuffer

import numpy as np
import matplotlib.pyplot as plt
import torch
import time


# ============================================================
# CONFIGURATION FLAGS
# ============================================================

TRAIN_WITH_GUI = False      # True → visualize training (slow)
EVAL_WITH_GUI = True        # True → visualize evaluation
EPISODES = 500
MAX_STEPS = 200

# Target real-time evaluation duration (seconds)
TARGET_EVAL_DURATION = 60
EVAL_DELAY = TARGET_EVAL_DURATION / MAX_STEPS


# ============================================================
# Fixed Baseline Controller
# ============================================================

def run_fixed_baseline(env, steps=200):

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

        # Reverse delta waiting reward
        total_waiting += -reward
        counter += 1

    return total_waiting


# ============================================================
# DQN Evaluation Mode (Real-Time GUI)
# ============================================================

def run_dqn_evaluation(agent):

    print("\nLaunching SUMO for DQN evaluation...")

    eval_env = TrafficEnv(gui=EVAL_WITH_GUI)
    eval_env.start()

    state = eval_env.get_state()
    total_waiting = 0
    done = False

    # Pure exploitation
    agent.epsilon = 0.0

    while not done:

        action = agent.select_action(state)
        next_state, reward, done = eval_env.step(action)

        total_waiting += -reward
        state = next_state

        # Slow down for real-time visualization
        if EVAL_WITH_GUI:
            time.sleep(EVAL_DELAY)

    eval_env.close()
    return total_waiting


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Initialize Training Environment
    # --------------------------------------------------------
    env = TrafficEnv(gui=TRAIN_WITH_GUI)
    env.start()

    state_dim = len(env.get_state())
    action_dim = env.num_phases

    agent = DQNAgent(state_dim, action_dim)
    replay_buffer = ReplayBuffer(50000)

    episode_rewards = []

    # --------------------------------------------------------
    # Baseline Comparison
    # --------------------------------------------------------
    print("Running fixed baseline...")
    baseline_wait = run_fixed_baseline(env, steps=MAX_STEPS)
    print("Baseline total waiting:", round(baseline_wait, 2))
    print("-" * 50)

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    print("Starting DQN training...\n")

    for episode in range(EPISODES):

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
            f"Episode {episode+1}/{EPISODES} | "
            f"Reward: {round(total_reward, 2)} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    env.close()
    print("\nTraining completed.")

    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------
    torch.save(agent.policy_net.state_dict(), "dqn_model.pth")
    print("Model saved as dqn_model.pth")

    # --------------------------------------------------------
    # Evaluation Mode
    # --------------------------------------------------------
    dqn_wait = run_dqn_evaluation(agent)

    print("\n------------------ FINAL RESULTS ------------------")
    print("Baseline total waiting:", round(baseline_wait, 2))
    print("DQN total waiting:", round(dqn_wait, 2))
    print("Average reward (last 50 episodes):",
          round(np.mean(episode_rewards[-50:]), 3))
    print("---------------------------------------------------")

    # --------------------------------------------------------
    # Plot Training Performance
    # --------------------------------------------------------
    plt.figure(figsize=(10, 5))

    plt.plot(episode_rewards, label="Raw Reward")

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