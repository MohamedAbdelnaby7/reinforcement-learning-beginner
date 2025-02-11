import numpy as np
import matplotlib.pyplot as plt
import time

# World size and parameters
WORLD_HEIGHT = 15
WORLD_WIDTH = 51
GAMMA = 0.95       # Discount factor
THETA = 1e-4       # Convergence threshold

# Actions (N, S, E, W, NE, NW, SE, SW)
ACTIONS = [(-1, 0), (1, 0), (0, 1), (0, -1),
           (-1, 1), (-1, -1), (1, 1), (1, -1)]
ACTION_NAMES = ['↑', '↓', '→', '←', '↗', '↖', '↘', '↙']

# Rewards
REWARD_GOAL = 100.0
REWARD_OBSTACLE = -50.0
REWARD_MOVE = -1.0

def create_world():
    """Initialize the 15x51 world with obstacles (1), free space (0), and place a goal (2)."""
    world = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ], dtype=int)
    # Place the goal at row=8, col=11 in 1-based => (7,10) in 0-based
    world[7, 10] = 2  # Mark this cell as the goal
    return world

def initialize_values(world):
    """Initialize a float array for the value function."""
    return np.zeros_like(world, dtype=float)

def initialize_policy(world):
    """
    Initialize a random deterministic policy for free cells (0).
    For obstacles (1) or goal (2), set policy to -1 (unused).
    """
    height, width = world.shape
    policy = -1 * np.ones((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            if world[i, j] == 0:  # free space
                policy[i, j] = np.random.randint(len(ACTIONS))
    return policy

def one_step_lookahead(i, j, action_idx, values, world):
    """
    Returns the expected (immediate reward + gamma * next_value) for
    executing 'action_idx' from state (i, j), under perfect action execution.
    If the next state is out of bounds or an obstacle, returns penalty accordingly.
    If it's the goal, returns REWARD_GOAL with no further value.
    Otherwise, returns REWARD_MOVE + gamma * next state's value.
    """
    height, width = world.shape
    di, dj = ACTIONS[action_idx]
    next_i, next_j = i + di, j + dj

    # Out of bounds => treat as obstacle
    if not (0 <= next_i < height and 0 <= next_j < width):
        return REWARD_OBSTACLE

    cell_type = world[next_i, next_j]
    if cell_type == 1:
        return REWARD_OBSTACLE
    elif cell_type == 2:  # goal
        return REWARD_GOAL
    else:
        # Free cell
        return REWARD_MOVE + GAMMA * values[next_i, next_j]

def policy_evaluation(policy, values, world):
    """Evaluate the current deterministic policy until convergence."""
    height, width = world.shape
    while True:
        delta = 0
        new_values = values.copy()
        for i in range(height):
            for j in range(width):
                # Only update free cells (0); skip obstacles (1) and goal (2)
                if world[i, j] != 0:
                    continue
                v_old = values[i, j]
                a = policy[i, j]
                new_values[i, j] = one_step_lookahead(i, j, a, values, world)
                delta = max(delta, abs(v_old - new_values[i, j]))
        values = new_values
        if delta < THETA:
            break
    return values

def policy_improvement(policy, values, world):
    """Improve the policy greedily given the current value function."""
    height, width = world.shape
    stable = True
    for i in range(height):
        for j in range(width):
            if world[i, j] != 0:
                continue
            old_action = policy[i, j]
            # Evaluate all possible actions
            action_values = [
                one_step_lookahead(i, j, a_idx, values, world)
                for a_idx in range(len(ACTIONS))
            ]
            best_action = np.argmax(action_values)
            policy[i, j] = best_action
            if best_action != old_action:
                stable = False
    return policy, stable

def policy_iteration(world):
    """Run iterative policy evaluation + improvement until stable."""
    values = initialize_values(world)
    policy = initialize_policy(world)

    while True:
        values = policy_evaluation(policy, values, world)
        policy, stable = policy_improvement(policy, values, world)
        if stable:
            break
    return policy, values

def value_iteration(world):
    """
    Perform value iteration until convergence.
    Returns (values, policy).
      - values: final value function
      - policy: derived deterministic policy from the final values
    """
    height, width = world.shape
    values = initialize_values(world)

    while True:
        delta = 0
        new_values = values.copy()
        for i in range(height):
            for j in range(width):
                if world[i, j] != 0:
                    continue
                old_val = values[i, j]
                # For each action, compute expected value
                action_values = [
                    one_step_lookahead(i, j, a_idx, values, world)
                    for a_idx in range(len(ACTIONS))
                ]
                best_val = max(action_values)
                new_values[i, j] = best_val
                delta = max(delta, abs(old_val - best_val))
        values = new_values
        if delta < THETA:
            break

    # Compute a greedy policy from the final values
    policy = -1 * np.ones((height, width), dtype=int)
    for i in range(height):
        for j in range(width):
            if world[i, j] == 0:
                action_values = [
                    one_step_lookahead(i, j, a_idx, values, world)
                    for a_idx in range(len(ACTIONS))
                ]
                best_a = np.argmax(action_values)
                policy[i, j] = best_a
    return np.array(policy, dtype=int), np.array(values)

def plot_policy_arrows(ax, policy, world, step=2):
    """
    Draw disconnected arrows for free cells.
      - step=2 means we skip every other cell to reduce visual clutter.
      - No arrows are drawn for obstacles or the goal.
    """
    height, width = world.shape

    # Collect points and vectors
    Xs, Ys, Us, Vs = [], [], [], []
    for i in range(0, height, step):
        for j in range(0, width, step):
            if world[i, j] == 0:  # free cell
                a_idx = policy[i, j]
                if a_idx >= 0:  # valid policy action
                    di, dj = ACTIONS[a_idx]
                    Xs.append(j)
                    Ys.append(i)
                    # Quiver: U = movement in x, V = movement in y (screen coords)
                    Us.append(dj)
                    Vs.append(di)  # negative because y goes down as i increases

    # Convert to arrays
    Xs = np.array(Xs)
    Ys = np.array(Ys)
    Us = np.array(Us)
    Vs = np.array(Vs)

    # Draw quiver
    ax.quiver(Xs, Ys, Us, Vs, color='black', pivot='mid',
              scale_units='xy', scale=1, headwidth=3, headlength=4)

def plot_results(policy, values, world):
    """Visualize the value function, policy arrows, and the world."""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot the value function as a background
    cax = ax.imshow(values, cmap='coolwarm', interpolation='none')
    fig.colorbar(cax, ax=ax, label='Value')

    # Overlay the policy arrows
    plot_policy_arrows(ax, policy, world, step=1)

    # Mark obstacles in black, goal in green
    # We can do this by overlaying with scatter or by adjusting the colormap
    # For a quick fix, let's scatter for obstacles and goal:
    obs_y, obs_x = np.where(world == 1)
    ax.scatter(obs_x, obs_y, color='k', marker='s', s=10, label='Obstacle')
    goal_y, goal_x = np.where(world == 2)
    ax.scatter(goal_x, goal_y, color='g', marker='*', s=100, label='Goal')

    ax.set_title('Value Function + Policy Arrows')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.invert_yaxis()  # So row 0 is at top visually
    ax.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

def main():
    world = create_world()
    start_time = time.time()
    policy, values = policy_iteration(world)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Policy iteration took {elapsed:.4f} seconds.")
    plot_results(policy, values, world)

    start_time = time.time()
    policy, values = value_iteration(world)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Value iteration took {elapsed:.4f} seconds.")
    plot_results(policy, values, world)

if __name__ == "__main__":
    main()