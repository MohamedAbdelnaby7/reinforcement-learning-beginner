import numpy as np
import matplotlib.pyplot as plt
import time
# policy hyperparameters
GAMMA = 0.8      # Discount factor
THETA = 1e-2       # Convergence threshold

# Actions (N, S, E, W, NE, NW, SE, SW)
ACTIONS = [
    (0, 1),   # 0 → Right (East)  
    (-1, 1),  # 1 ↗ Up-Right (North-East)  
    (-1, 0),  # 2 ↑ Up (North)  
    (-1, -1), # 3 ↖ Up-Left (North-West)  
    (0, -1),  # 4 ← Left (West)  
    (1, -1),  # 5 ↙ Down-Left (South-West)  
    (1, 0),   # 6 ↓ Down (South)  
    (1, 1)    # 7 ↘ Down-Right (South-East)  
]

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
0
def compute_action_probabilities(action_idx, transition_model):
    """Compute action probabilities for a given action index based on the transition model."""
    return transition_model[action_idx,:]

def create_transition_model(ACTIONS):
    """Generate a transition model where actions deviate by ±45° with 10% probability."""
    num_actions = len(ACTIONS)
    transition_model = np.zeros((num_actions, num_actions))

    for a in range(num_actions):
        transition_model[a, a] = 0.8  # Intended action
        transition_model[a, (a - 1) % num_actions] = 0.1  # -45° (counterclockwise)
        transition_model[a, (a + 1) % num_actions] = 0.1  # +45° (clockwise)

    return transition_model
    
def one_step_lookahead(i, j, action_idx, values, world, transition_model=None):
    """
    Returns the expected (immediate reward + gamma * next_value) for
    executing 'action_idx' from state (i, j), under perfect action execution if all probabilities other than action is zero.
    otherwise compute expected value of taking action_idx from (i, j) using given probabilities.
    If the next state is out of bounds or an obstacle, returns penalty accordingly.
    If it's the goal, returns REWARD_GOAL with no further value.
    Otherwise, returns REWARD_MOVE + gamma * next state's value.
    """
    height, width = world.shape
    if transition_model is None:
        # Directly compute if the action execution is deterministic
        di, dj = ACTIONS[action_idx]
        next_i, next_j = i + di, j + dj
        if not (0 <= next_i < height and 0 <= next_j < width):
            return REWARD_OBSTACLE
        cell_type = world[next_i, next_j]
        if cell_type == 1:
            return REWARD_OBSTACLE
        elif cell_type == 2:
            return REWARD_GOAL
        else:
            return REWARD_MOVE + GAMMA * values[next_i, next_j]
    else:
        action_probabilities = compute_action_probabilities(action_idx, transition_model)
        expected_value = 0.0
        for a_idx, prob in enumerate(action_probabilities):
            di, dj = ACTIONS[a_idx]
            next_i, next_j = i + di, j + dj
            if not (0 <= next_i < height and 0 <= next_j < width):
                reward = REWARD_OBSTACLE
                next_val = 0.0
            else:
                cell_type = world[next_i, next_j]
                if cell_type == 1:
                    reward = REWARD_OBSTACLE
                    next_val = 0.0
                elif cell_type == 2:
                    reward = REWARD_GOAL
                    next_val = 0.0
                else:
                    reward = REWARD_MOVE
                    next_val = values[next_i, next_j]
            expected_value += prob * (reward + GAMMA * next_val)
        return expected_value

def policy_evaluation(policy, values, world, transition_model=None):
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
                new_values[i, j] = one_step_lookahead(i, j, a, values, world, transition_model)
                delta = max(delta, abs(v_old - new_values[i, j]))
        values = new_values
        if delta < THETA:
            break
    return values

def policy_improvement(policy, values, world, transition_model=None):
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
                one_step_lookahead(i, j, a_idx, values, world, transition_model)
                for a_idx in range(len(ACTIONS))
            ]
            best_action = np.argmax(action_values)
            policy[i, j] = best_action
            if best_action != old_action:
                #print(f"Policy change at ({i}, {j}): {old_action} → {best_action}, ΔV={action_values[best_action] - action_values[old_action]}")
                stable = False
    return policy, stable

def policy_iteration(world, transition_model=None):
    """Run iterative policy evaluation + improvement until stable."""
    values = initialize_values(world)
    policy = initialize_policy(world)

    while True:
        values = policy_evaluation(policy, values, world)
        policy, stable = policy_improvement(policy, values, world, transition_model)
        if stable:
            break
    return policy, values

def value_iteration(world, transition_model=None):
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
                    one_step_lookahead(i, j, a_idx, values, world, transition_model)
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
def generalized_policy_iteration(world, transition_model=None):
    """Perform Generalized Policy Iteration (GPI) where evaluation and improvement happen iteratively."""
    values = initialize_values(world)    
    height, width = world.shape
    policy = initialize_policy(world)

    while True:
        delta = 0
        new_values = values.copy()
        for i in range(height):
            for j in range(width):
                if world[i, j] != 0:
                    continue
                new_values[i, j] = one_step_lookahead(i, j, policy[i, j], values, world, transition_model)
        values = new_values
        
        stable = True
        for i in range(height):
            for j in range(width):
                if world[i, j] != 0:
                    continue
                old_action = policy[i, j]
                action_values = [one_step_lookahead(i, j, a_idx, values, world, transition_model) for a_idx in range(len(ACTIONS))]
                best_a = np.argmax(action_values)
                policy[i, j] = best_a
                if best_a != old_action:
                    stable = False
        
        if stable and delta < THETA:
            break
    return policy, values

def plot_policy_arrows(ax, policy, world, step=1):
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

    world = create_world()
    start_time = time.time()
    transition_model = create_transition_model(ACTIONS)
    policy, values = policy_iteration(world, transition_model)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Policy iteration with uncertainty took {elapsed:.4f} seconds.")
    plot_results(policy, values, world)

    start_time = time.time()
    policy, values = value_iteration(world)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Value iteration took {elapsed:.4f} seconds.")
    plot_results(policy, values, world)

    start_time = time.time()
    policy, values = value_iteration(world, transition_model)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Value iteration with uncertainty took {elapsed:.4f} seconds.")
    plot_results(policy, values, world)

    start_time = time.time()
    policy, values = generalized_policy_iteration(world)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"GPI took {elapsed:.4f} seconds.")
    plot_results(policy, values, world)

    start_time = time.time()
    policy, values = generalized_policy_iteration(world, transition_model)
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"GPI with uncertainty took {elapsed:.4f} seconds.")
    plot_results(policy, values, world)

if __name__ == "__main__":
    main()