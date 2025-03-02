import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Grid constants
ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)
CLIFF = [(3, c) for c in range(1, 11)]

# Hyperparameters
ALPHA = 0.5
GAMMA = 1.0
EPISODES = 500
REPEATS = 100  # Number of runs to average over
EPSILON = 0.1
MIN_EPSILON = 0.01
DECAY_RATE = 0.999

# Actions: Up, Down, Left, Right
ACTION_DICT = {
    0: (-1, 0),  # UP
    1: (1, 0),   # DOWN
    2: (0, -1),  # LEFT
    3: (0, 1)    # RIGHT
}
ACTION_NAMES = ['↑', '↓', '←', '→']  # For policy visualization

def step(state, action):
    """
    Execute one step in the Cliff environment.
    Returns next_state, reward, done.
    """
    r, c = state
    dr, dc = ACTION_DICT[action]
    nr, nc = r + dr, c + dc

    # Bound within grid
    nr = max(0, min(ROWS - 1, nr))
    nc = max(0, min(COLS - 1, nc))

    # Check cliff
    if (nr, nc) in CLIFF:
        return START, -100, True

    # Check goal
    if (nr, nc) == GOAL:
        return (nr, nc), 0, True

    # Otherwise
    return (nr, nc), -1, False

def epsilon_greedy(Q, state, epsilon):
    """
    Epsilon-greedy action selection from Q.
    """
    if np.random.rand() < epsilon:
        return np.random.choice(list(ACTION_DICT.keys()))
    else:
        return np.argmax(Q[state])

def sarsa_run(episodes, alpha, gamma, eps, decay=False, seed=None):
    """
    Single run of SARSA (with optional decay).
    Returns the list of episode rewards.
    """
    if seed is not None:
        np.random.seed(seed)

    Q = defaultdict(lambda: np.zeros(len(ACTION_DICT)))
    rewards = []
    for ep in range(episodes):
        state = START
        action = epsilon_greedy(Q, state, eps)
        ep_reward = 0
        done = False
        while not done:
            next_state, reward, done = step(state, action)
            ep_reward += reward
            if not done:
                next_action = epsilon_greedy(Q, next_state, eps)
            else:
                next_action = 0  # doesn't matter if terminal

            # SARSA update
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state = next_state
            action = next_action

        if decay:
            eps = max(MIN_EPSILON, eps * DECAY_RATE)
        rewards.append(ep_reward)
    return Q, rewards

def qlearning_run(episodes, alpha, gamma, eps, decay=False, seed=None):
    """
    Single run of Q-learning (with optional decay).
    Returns the list of episode rewards.
    """
    if seed is not None:
        np.random.seed(seed)

    Q = defaultdict(lambda: np.zeros(len(ACTION_DICT)))
    rewards = []
    for ep in range(episodes):
        state = START
        ep_reward = 0
        done = False
        while not done:
            action = epsilon_greedy(Q, state, eps)
            next_state, reward, done = step(state, action)
            ep_reward += reward

            # Off-policy update
            best_next_action = np.argmax(Q[next_state])
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
            state = next_state

        if decay:
            eps = max(MIN_EPSILON, eps * DECAY_RATE)
        rewards.append(ep_reward)
    return Q, rewards

def average_runs(agent_fn, episodes=EPISODES, alpha=ALPHA, gamma=GAMMA, eps=EPSILON,
                 decay=False, repeats=REPEATS):
    """
    Averages the reward over multiple runs (different seeds).
    agent_fn: a function that runs a single agent (SARSA or Q-learning).
    returns: (avg_rewards, final_Q)
    """
    all_rewards = np.zeros((repeats, episodes))
    Q_final = None
    for r in range(repeats):
        seed = r  # or any random assignment
        Q, rewards = agent_fn(episodes, alpha, gamma, eps, decay, seed=seed)
        all_rewards[r, :] = rewards
        if r == repeats - 1:
            Q_final = Q  # keep the last Q
    avg_rewards = np.mean(all_rewards, axis=0)
    return avg_rewards, Q_final

def extract_policy(Q):
    """
    Converts Q (dict of arrays) into a 2D array of best actions for each state in the grid.
    We'll return a 2D array policy where policy[row, col] = best_action_index.
    """
    policy_grid = np.full((ROWS, COLS), -1, dtype=int)
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) == GOAL or (r, c) in CLIFF:
                continue
            policy_grid[r, c] = np.argmax(Q[(r, c)])
    return policy_grid

def plot_policy(ax, policy_grid):
    """
    Visualize the policy as arrows on a given axis 'ax'.
    For the cliff area, we show 'X', for the goal 'G'.
    """
    ax.set_xlim(-0.5, COLS - 0.5)
    ax.set_ylim(ROWS - 0.5, -0.5)
    ax.set_xticks(range(COLS))
    ax.set_yticks(range(ROWS))
    ax.grid(True)

    # Plot cliff and goal
    for (r, c) in CLIFF:
        ax.text(c, r, 'X', ha='center', va='center', color='red', fontsize=14, fontweight='bold')
    gr, gc = GOAL
    ax.text(gc, gr, 'G', ha='center', va='center', color='green', fontsize=14, fontweight='bold')

    # Plot policy arrows
    for r in range(ROWS):
        for c in range(COLS):
            if (r, c) in CLIFF or (r, c) == GOAL or (r, c) == START:
                continue
            action = policy_grid[r, c]
            if action == -1:
                continue
            ax.text(c, r, ACTION_NAMES[action], ha='center', va='center', color='blue', fontsize=10)

    # Mark the start
    sr, sc = START
    ax.text(sc, sr, 'S', ha='center', va='center', color='orange', fontsize=12, fontweight='bold')
    ax.set_title("Policy Visualization")

def main():
    # 1) SARSA vs Q-learning with fixed epsilon
    sarsa_rewards_fixed, sarsaQ_fixed = average_runs(sarsa_run, decay=False)
    qlearning_rewards_fixed, qlearnQ_fixed = average_runs(qlearning_run, decay=False)

    # 2) SARSA vs Q-learning with decaying epsilon
    sarsa_rewards_decay, sarsaQ_decay = average_runs(sarsa_run, decay=True)
    qlearning_rewards_decay, qlearnQ_decay = average_runs(qlearning_run, decay=True)

    # Convert Q to 2D policy arrays
    sarsa_pol_fixed = extract_policy(sarsaQ_fixed)
    qlearn_pol_fixed = extract_policy(qlearnQ_fixed)
    sarsa_pol_decay = extract_policy(sarsaQ_decay)
    qlearn_pol_decay = extract_policy(qlearnQ_decay)

    # Create a 2x4 figure => 4 subplots horizontally, 2 rows
    # But for clarity, let's do 2 subplots per row => total 4 subplots
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(18, 8))

    # Row 1, col 1 => SARSA fixed (rewards)
    axs[0, 0].plot(sarsa_rewards_fixed, label='SARSA, fixed eps')
    axs[0, 0].set_title("SARSA (fixed eps) Average Rewards")
    axs[0, 0].set_xlabel("Episode")
    axs[0, 0].set_ylabel("Avg Reward")
    axs[0, 0].grid(True)

    # Row 1, col 2 => SARSA fixed policy
    plot_policy(axs[0, 1], sarsa_pol_fixed)

    # Row 1, col 3 => Q-learning fixed (rewards)
    axs[0, 2].plot(qlearning_rewards_fixed, color='orange', label='Q-learning, fixed eps')
    axs[0, 2].set_title("Q-learning (fixed eps) Average Rewards")
    axs[0, 2].set_xlabel("Episode")
    axs[0, 2].set_ylabel("Avg Reward")
    axs[0, 2].grid(True)

    # Row 1, col 4 => Q-learning fixed policy
    plot_policy(axs[0, 3], qlearn_pol_fixed)

    # Row 2, col 1 => SARSA decay (rewards)
    axs[1, 0].plot(sarsa_rewards_decay, label='SARSA, decaying eps')
    axs[1, 0].set_title("SARSA (decaying eps) Average Rewards")
    axs[1, 0].set_xlabel("Episode")
    axs[1, 0].set_ylabel("Avg Reward")
    axs[1, 0].grid(True)

    # Row 2, col 2 => SARSA decay policy
    plot_policy(axs[1, 1], sarsa_pol_decay)

    # Row 2, col 3 => Q-learning decay (rewards)
    axs[1, 2].plot(qlearning_rewards_decay, color='orange', label='Q-learning, decaying eps')
    axs[1, 2].set_title("Q-learning (decaying eps) Average Rewards")
    axs[1, 2].set_xlabel("Episode")
    axs[1, 2].set_ylabel("Avg Reward")
    axs[1, 2].grid(True)

    # Row 2, col 4 => Q-learning decay policy
    plot_policy(axs[1, 3], qlearn_pol_decay)

    fig.suptitle("Cliff Walking: SARSA vs Q-learning (Average of Multiple Runs)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()
