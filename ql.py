# Credit: Q-Learning system by Dr. Daniel Soper - https://youtu.be/iKdlKYG78j4 adapted for FTC's Into the Deep season by Rishab Shah and Kaushik Chamchani (14162 Bots & Bytes)

"""
This is a Q-Learning system used to figure out how good it is to be at any given position on the Into the Deep board is. We have first defined the board,
using arbitrary values to describe how we feel certain position are from the perspective of blue (ex. blue tower is very good, red tower is very bad). Then,
we run a simulation 1,000,000,000 times of an AI agent that gets punished when it does certain actions (strays away from the blue side, goes into the red tower, etc.),
and rewarded when it does something good (go to blue tower, go to the blue human sample area, etc.). The AI agents then learn from these rewards and punishments,
improving over time and, using a mathematical formula, assign a value to each action possible (up, down, left, right) in each position on the board. We then average 
all of these values to get a representation of how good it is to be in a certain position on the board.
"""

import numpy as np
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numba import njit

environment_rows = 12
environment_columns = 12

# define actions
# numeric action codes: 0 = up, 1 = right, 2 = down, 3 = left
actions = ("up", "right", "down", "left")

# Create a 2D numpy array to hold the rewards for each state.
# The array contains 12 rows and 12 columns (to match the shape of the environment), and each value is initialized to -1.00
rewards = np.full((environment_rows, environment_columns), -1.0)
# blue human player zone
rewards[0, 0] = 25.0
rewards[0, 1] = 25.0
rewards[0, 2] = 25.0
# blue tower
rewards[0, 10] = 100.0
rewards[0, 11] = 100.0
rewards[1, 11] = 100.0
# blue stripes
rewards[3, 0] = 45.0
rewards[3, 1] = 45.0
# yellow stripes close to blue
rewards[3, 10] = 55.0
rewards[3, 11] = 55.0
# specimen score area
rewards[3, 5] = 87.0
rewards[3, 6] = 87.0
# inside specimen area
rewards[4, 5] = -100.0
rewards[4, 6] = -100.0
rewards[5, 5] = -100.0
rewards[5, 6] = -100.0
rewards[6, 5] = -100.0
rewards[6, 6] = -100.0
rewards[7, 5] = -100.0
rewards[7, 6] = -100.0
# red ascent zone
rewards[4, 4] = 45.0
rewards[5, 4] = 45.0
rewards[6, 4] = 45.0
rewards[7, 4] = 45.0
# blue ascent zone
rewards[4, 7] = 55.0
rewards[5, 7] = 55.0
rewards[6, 7] = 55.0
rewards[7, 7] = 55.0
# yellow stripes on red side
rewards[8, 0] = 15.0
rewards[8, 1] = 15.0
# red stripes
rewards[8, 10] = -90.0
rewards[8, 11] = -90.0
# red tower
rewards[10, 0] = -100.0
rewards[11, 0] = -100.0
rewards[11, 1] = -100.0
# red human player zone
rewards[11, 9] = -100.0
rewards[11, 10] = -100.0
rewards[11, 11] = -100.0

# make it less valuable for blue as the map goes closer to red
for y in range(0, 12):
    for x in range(0, 12):
        rewards[y][x] = rewards[y][x] - y

# define a function that determines if the specified location is a terminal state
@njit(fastmath=True)
def is_terminal_state(current_row_index, current_column_index):
    # if the reward for this location is -12.0<=x<0.0 (because of the cascading rewards for being closer to red)
    if (
        -12.0 <= rewards[current_row_index, current_column_index] < 0.0
    ):
        return False
    else:
        return True


# define a function that will choose a random, non-terminal starting location
@njit(fastmath=True)
def get_starting_location():
    while True:
        row, col = np.random.randint(environment_rows), np.random.randint(
            environment_columns
        )
        if not is_terminal_state(row, col):
            return row, col


# define an epsilon greedy algorithm that will choose which action to take next (i.e., where to move next)
@njit(fastmath=True)
def get_next_action(current_row_index, current_column_index, epsilon, q_values):
    # if a randomly chosen value between 0 and 1 is less than epsilon,
    # then choose the most promising value from the Q-table for this state.
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_row_index, current_column_index])
    return np.random.randint(4)


# define a function that will get the next location based on the chosen action
@njit(fastmath=True)
def get_next_location(current_row_index, current_column_index, action_index, actions):
    new_row_index, new_column_index = current_row_index, current_column_index
    if actions[action_index] == "up" and current_row_index > 0:
        new_row_index -= 1
    elif (
        actions[action_index] == "right"
        and current_column_index < environment_columns - 1
    ):
        new_column_index += 1
    elif actions[action_index] == "down" and current_row_index < environment_rows - 1:
        new_row_index += 1
    elif actions[action_index] == "left" and current_column_index > 0:
        new_column_index -= 1
    return new_row_index, new_column_index

@njit(fastmath=True)
def train_agent_episode(q_values, epsilon, discount_factor, learning_rate, actions):
    # get the starting location for this episode
    row_index, column_index = get_starting_location()
    experiences = [] # Store experiences: (s, a, r, s')

    # continue taking actions (i.e., moving) until we reach a terminal state
    while not is_terminal_state(row_index, column_index):
        # choose which action to take (i.e., where to move next)
        action_index = get_next_action(row_index, column_index, epsilon, q_values)

        # perform the chosen action, and transition to the next state (i.e., move to the next location)
        old_row_index, old_column_index = row_index, column_index
        row_index, column_index = get_next_location(
            row_index, column_index, action_index, actions
        )
        reward = rewards[row_index, column_index]
        experiences.append((old_row_index, old_column_index, action_index, reward, row_index, column_index))

    # Batch update after the episode ends
    if experiences: # Only update if there are experiences
        for old_row_index, old_column_index, action_index, reward, row_index, column_index in experiences:
            old_q_value = q_values[old_row_index, old_column_index, action_index]
            temporal_difference = (
                    reward
                    + (discount_factor * np.max(q_values[row_index, column_index]))
                    - old_q_value
            )
            new_q_value = old_q_value + (learning_rate * temporal_difference)
            q_values[old_row_index, old_column_index, action_index] = new_q_value
    return q_values

def train_agent(episodes, epsilon, discount_factor, learning_rate, actions):
    qv = np.zeros((environment_rows, environment_columns, 4))

    for _ in tqdm(range(episodes)):
        qv = train_agent_episode(qv, epsilon, discount_factor, learning_rate, actions)

    return qv

# Example usage
if __name__ == '__main__':
    # define training parameters
    epsilon = 0.9  # the percentage of time when we should take the best action (instead of a random action)
    discount_factor = 0.9  # discount factor for future rewards
    learning_rate = 0.9  # the rate at which the AI agent should learn
    num_episodes = 1000000000  # the number of episodes the agent should undertake

    q_values = train_agent(num_episodes, epsilon, discount_factor, learning_rate, actions)
    print("Training complete!")


# print rewards matrix and averages all 4 possibilities
# we use mean rather than max because mean tells us the versatility of a position, rather than the best possible action in a given position
# initialize a list to collect all rows
data = []

# calculate mean values for each position and store them in q_matrix
for row in range(0, 12):
    prow = []
    for column in range(0, 12):
        prow.append(round(mean(q_values[row][column]), 1))
    data.append(prow)

# convert the list of lists into a 2D numpy array
data = np.array(data)

print(data)

# blue human player zone
data[0, 0] = 25.0
data[0, 1] = 25.0
data[0, 2] = 25.0
data[0, 10] = 100.0
data[0, 11] = 100.0
data[1, 11] = 100.0
data[3, 0] = 45.0
data[3, 1] = 45.0
data[3, 10] = 55.0
data[3, 11] = 55.0
data[3, 5] = 87.0
data[3, 6] = 87.0
data[4, 5] = -100.0
data[4, 6] = -100.0
data[5, 5] = -100.0
data[5, 6] = -100.0
data[6, 5] = -100.0
data[6, 6] = -100.0
data[7, 5] = -100.0
data[7, 6] = -100.0
data[4, 4] = 45.0
data[5, 4] = 45.0
data[6, 4] = 45.0
data[7, 4] = 45.0
data[4, 7] = 55.0
data[5, 7] = 55.0
data[6, 7] = 55.0
data[7, 7] = 55.0
data[8, 0] = 15.0
data[8, 1] = 15.0
data[8, 10] = -90.0
data[8, 11] = -90.0
data[10, 0] = -100.0
data[11, 0] = -100.0
data[11, 1] = -100.0
data[11, 9] = -100.0
data[11, 10] = -100.0
data[11, 11] = -100.0


# Set up figure and axis
fig, ax = plt.subplots(figsize=(12, 12), dpi=90)  # 1080x1080 pixels
ax.set_aspect("equal")

# Create a mask for zeros to manually set their color to black
masked_data = np.copy(data)
masked_data[masked_data == 0] = np.nan  # Set zero values to NaN for the colormap

# Display the grid with a reversed diverging colormap
cmap = plt.get_cmap(
    "seismic_r"
)  # Use reversed 'seismic' colormap for blue positive and red negative
norm = mcolors.TwoSlopeNorm(
    vmin=data.min(), vcenter=0, vmax=data.max()
)  # Normalize with center at 0

# Show the data as an image with masked zeros
cax = ax.matshow(masked_data, cmap=cmap, norm=norm)

# Overlay black cells where the original value was zero
ax.matshow(np.where(data == 0, 1, np.nan), cmap="gray", vmin=0, vmax=1, alpha=1)

# Add color bar for reference
fig.colorbar(cax)

# Add text annotations for each cell
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        if data[i, j] == 100.0:
            # Add text for other values
            ax.text(
                j,
                i,
                str(data[i, j]),
                va="center",
                ha="center",
                color="black",
            )
        else:
            # Add text for other values
            ax.text(
                j,
                i,
                str(data[i, j]),
                va="center",
                ha="center",
                color="white" if abs(data[i, j]) >= (data.max() / 2) else "black",
            )

# Remove ticks and labels
ax.set_xticks([])
ax.set_yticks([])

plt.show()
