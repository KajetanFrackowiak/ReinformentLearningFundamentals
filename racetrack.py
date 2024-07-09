import numpy as np
import random
import matplotlib.pyplot as plt

# Define constants
GRID_WIDTH = 10  # Example width, adjust as needed
GRID_HEIGHT = 10  # Example height, adjust as needed
MAX_VELOCITY = 4  # Max velocity in both x and y direction
DISCOUNT_FACTOR = 1.0  # No discounting
EPSILON = 0.1  # For epsilon-greedy policy

# Define racetrack as a 2D array where 0 is track, 1 is boundary, 2 is start, and 3 is finish
racetrack = np.zeros((GRID_HEIGHT, GRID_WIDTH))
# Initialize start and finish lines
start_positions = [(0, i) for i in range(GRID_WIDTH // 3)]
finish_positions = [
    (GRID_HEIGHT - 1, i) for i in range(2 * GRID_WIDTH // 3, GRID_WIDTH)
]

# Initialize Q-values and Returns
Q = {}
returns = {}
for x in range(GRID_HEIGHT):
    for y in range(GRID_WIDTH):
        for vx in range(MAX_VELOCITY + 1):
            for vy in range(MAX_VELOCITY + 1):
                for ax in [-1, 0, 1]:
                    for ay in [-1, 0, 1]:
                        state = ((x, y), (vx, vy))
                        action = (ax, ay)
                        Q[(state, action)] = 0.0
                        returns[(state, action)] = []


def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.choice([(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]])
    else:
        q_values = {
            action: Q[(state, action)]
            for action in [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
        }
        return max(q_values, key=q_values.get)


def take_action(state, action):
    (x, y), (vx, vy) = state
    ax, ay = action
    if random.uniform(0, 1) < 0.1:
        ax, ay = 0, 0  # Noise in action

    vx = min(max(vx + ax, 0), MAX_VELOCITY)
    vy = min(max(vy + ay, 0), MAX_VELOCITY)

    new_x = x + vx
    new_y = y + vy

    # Check if out of bounds or hitting boundary
    if (
        new_x >= GRID_HEIGHT
        or new_x < 0
        or new_y >= GRID_WIDTH
        or new_y < 0
        or racetrack[new_x, new_y] == 1
    ):
        new_x, new_y = random.choice(start_positions)
        vx, vy = 0, 0
        reward = -1
    elif (new_x, new_y) in finish_positions:
        reward = 0
    else:
        reward = -1

    return ((new_x, new_y), (vx, vy)), reward


def generate_episode(policy):
    episode = []
    state = random.choice(start_positions), (0, 0)
    while True:
        action = policy(state)
        new_state, reward = take_action(state, action)
        episode.append((state, action, reward))
        if new_state[0] in finish_positions:
            break
        state = new_state
    return episode


def monte_carlo_control():
    for episode_num in range(10000):  # Number of episodes
        episode = generate_episode(lambda s: choose_action(s, EPSILON))
        G = 0
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = reward + DISCOUNT_FACTOR * G
            if not any((state == x[0] and action == x[1]) for x in episode[:t]):
                returns[(state, action)].append(G)
                Q[(state, action)] = np.mean(returns[(state, action)])
        # Policy improvement
        policy = {}
        for s in Q:
            state = s[0]
            q_values = {
                action: Q[(state, action)]
                for action in [(ax, ay) for ax in [-1, 0, 1] for ay in [-1, 0, 1]]
            }
            policy[state] = max(q_values, key=q_values.get)
    return policy


def display_trajectory(policy):
    state = random.choice(start_positions), (0, 0)
    trajectory = [state]
    while state[0] not in finish_positions:
        action = policy[state]
        new_state, _ = take_action(state, action)
        trajectory.append(new_state)
        state = new_state

    # Extract positions for plotting
    positions = [pos for pos, vel in trajectory]
    xs, ys = zip(*positions)

    # Plot the trajectory
    plt.plot(ys, xs, marker="o")
    plt.xlim(0, GRID_WIDTH)
    plt.ylim(0, GRID_HEIGHT)
    plt.gca().invert_yaxis()  # Invert y-axis to match grid representation
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Car Trajectory")
    plt.show()


optimal_policy = monte_carlo_control()

for i in range(5):
    print(f"Trajectory {i + 1}:")
    display_trajectory(optimal_policy)
    print("\n")
