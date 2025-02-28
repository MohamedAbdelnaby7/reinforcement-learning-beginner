import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Environment Constants
ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)
CLIFF = [(3, c) for c in range(1, 11)]  # (3,1) to (3,10)

# Hyperparameters
ALPHA = 0.5      # Learning rate
GAMMA = 1.0      # Discount factor
EPISODES = 500   # Number of episodes
EPSILON = 0.1    # Initial exploration probability
MIN_EPSILON = 0.01
DECAY_RATE = 0.999  # Epsilon decay each episode

# Actions (Up, Down, Left, Right)
ACTIONS = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}

def step(state, action):
    """
    Take a step in the environment from 'state' using 'action'.
    Returns (next_state, reward, done).
    """
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc

    # Bound the next position within the grid
    nr = max(0, min(ROWS - 1, nr))
    nc = max(0, min(COLS - 1, nc))

    # Check if agent stepped off the cliff
    if (nr, nc) in CLIFF:
        return START, -100, True  # Falls off cliff => reset to start, done = True

    # Check if agent reached goal
    if (nr, nc) == GOAL:
        return (nr, nc), 0, True  # Reached goal => done = True, reward = 0

    # Otherwise normal step
    return (nr, nc), -1, False

def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon-greedy policy derived from Q.
    Returns an action index.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(list(ACTIONS.keys()))
    else:
        return np.argmax(Q[state])

def sarsa_train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, eps=EPSILON, 
                decay=False):
    """
    Train agent using SARSA.
    Returns:
        Q: The action-value function (dict of dict)
        rewards: List of sums of rewards per episode
    """
    # Q is a dict of dicts: Q[state][action] = value
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    rewards = []

    for ep in range(episodes):
        state = START
        done = False
        # Choose action using epsilon-greedy
        action = epsilon_greedy(Q, state, eps)
        ep_reward = 0

        while not done:
            next_state, reward, done = step(state, action)
            ep_reward += reward

            # Choose next action from next state
            if not done:
                next_action = epsilon_greedy(Q, next_state, eps)
            else:
                next_action = 0  # doesn't matter if done

            # SARSA update
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            state = next_state
            action = next_action
        
        # Decay epsilon if requested
        if decay:
            eps = max(MIN_EPSILON, eps * DECAY_RATE)

        rewards.append(ep_reward)

    return Q, rewards

def qlearning_train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, eps=EPSILON, 
                    decay=False):
    """
    Train agent using Q-learning.
    Returns:
        Q: The action-value function (dict of dict)
        rewards: List of sums of rewards per episode
    """
    Q = defaultdict(lambda: np.zeros(len(ACTIONS)))
    rewards = []

    for ep in range(episodes):
        state = START
        done = False
        ep_reward = 0

        while not done:
            action = epsilon_greedy(Q, state, eps)
            next_state, reward, done = step(state, action)
            ep_reward += reward

            # Q-learning update
            best_next_action = np.argmax(Q[next_state])  # greedy wrt Q
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])

            state = next_state
        
        # Decay epsilon if requested
        if decay:
            eps = max(MIN_EPSILON, eps * DECAY_RATE)

        rewards.append(ep_reward)

    return Q, rewards

def run_experiments():
    """
    Run SARSA and Q-Learning with:
      1) Fixed epsilon
      2) Decaying epsilon
    Plot results.
    """
    # ============ 1) SARSA vs Q-learning (fixed epsilon) ============
    sarsa_Q, sarsa_rewards = sarsa_train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, eps=EPSILON, decay=False)
    qlearn_Q, qlearn_rewards = qlearning_train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, eps=EPSILON, decay=False)

    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_rewards, label='SARSA (fixed eps)')
    plt.plot(qlearn_rewards, label='Q-learning (fixed eps)')
    plt.title("Episode Rewards with Fixed Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ============ 2) SARSA vs Q-learning (decaying epsilon) ============
    sarsa_Q_decay, sarsa_rewards_decay = sarsa_train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, eps=EPSILON, decay=True)
    qlearn_Q_decay, qlearn_rewards_decay = qlearning_train(episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, eps=EPSILON, decay=True)

    plt.figure(figsize=(10, 6))
    plt.plot(sarsa_rewards_decay, label='SARSA (decaying eps)')
    plt.plot(qlearn_rewards_decay, label='Q-learning (decaying eps)')
    plt.title("Episode Rewards with Decaying Epsilon")
    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return final Q and rewards if needed
    return (sarsa_Q, sarsa_rewards), (qlearn_Q, qlearn_rewards), \
           (sarsa_Q_decay, sarsa_rewards_decay), (qlearn_Q_decay, qlearn_rewards_decay)

def main():
    # Run the experiments and generate plots
    run_experiments()
    print("Experiments completed. Plots shown.")

if __name__ == "__main__":
    main()
