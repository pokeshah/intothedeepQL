# intothedeepQL
An exploration into strategy derived from table-based Q-Learning agents

This is a Q-Learning system used to figure out how good it is to be at any given position on the Into the Deep board is. We have first defined the board, using arbitrary values to describe how we feel certain position are from the perspective of blue (ex. blue tower is very good, red tower is very bad). Then, we run a simulation 1,000,000,000 times of an AI agent that gets punished when it does certain actions (strays away from the blue side, goes into the red tower, etc.), and rewarded when it does something good (go to blue tower, go to the blue human sample area, etc.). The AI agents then learn from these rewards and punishments, improving over time and, using a mathematical formula, assign a value to each action possible (up, down, left, right) in each position on the board. We then average all of these values to get a representation of how good it is to be in a certain position on the board.

TODO:
- Simulate the actual game instead of using predefined values
- Factor in distance better using acceleration
- Swap to Deep QL for a better representation
- ~~Use Numba to achieve faster episode times~~

Credit: Q-Learning system by Dr. Daniel Soper - https://youtu.be/iKdlKYG78j4
