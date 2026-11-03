# Implementation-of-Value-Iteration-for-Optimal-Policy-Computation-using-Gymnasium

---
## Aim

To implement the **Value Iteration** algorithm for solving a finite Markov Decision Process using the Gymnasium `FrozenLake-v1` environment, and to compute the optimal state-value function and optimal policy using the Bellman optimality equation.

---

## Problem Statement






## Software Requirements

## Environment Description


## MDP Representation


## Theory




## Algorithm





## Python Program

```
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Create FrozenLake Environment
# -------------------------------------------------
env_desc = [

    "FFHF",
    "FSFF",
    "FHFH",
    "GFFF"
]


env = gym.make(
    "FrozenLake-v1",
    desc=env_desc,
    is_slippery=True
)



---

```
# -------------------------------------------------
# Value Iteration Algorithm
# -------------------------------------------------

def value_iteration(env, gamma=0.99, theta=1e-8):

    P = env.unwrapped.P

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    V = np.zeros(n_states)
    policy = np.zeros(n_states, dtype=int)

    iteration = 0

    while True:
        delta = 0

        for state in range(n_states):

            action_values = []

            for action in range(n_actions):

                value = 0

                for probability, next_state, reward, done in P[state][action]:

                    value += probability * (
                        reward + gamma * V[next_state] * (not done)
                    )

                action_values.append(value)

            best_value = max(action_values)

            delta = max(delta, abs(best_value - V[state]))

            V[state] = best_value

            # Update optimal policy
            policy[state] = np.argmax(action_values)

        iteration += 1

        if delta < theta:
            break

    return V, policy, iteration
    ```
    
    ```
    
# -------------------------------------------------
# Display Output
# -------------------------------------------------
```
print("Name: Nithish kumar s")
print("Register Number: 212224230190")
print("Value Iteration Completed")

print("Number of Iterations:", iterations)

print("\nOptimal State-Value Function:")
print(np.round(V.reshape(4, 4), 4))

action_symbols = {
    0: "L",
    1: "D",
    2: "R",
    3: "U"
}

policy_grid = np.array(
    [action_symbols[action] for action in policy]
).reshape(4, 4)

print("\nOptimal Policy:")
print(policy_grid)

print("\nGrid Layout:")

for row in env.unwrapped.desc:
    print(" ".join(cell.decode("utf-8") for cell in row))

env.close()

```

## Output

```
<img width="720" height="601" alt="image" src="https://github.com/user-attachments/assets/310b07b0-662d-46d8-b3a2-7e92868cea2d" />

---
```
## Result
```
W4. Final Result for Record
Result: Thus, the Value Iteration algorithm was successfully implemented using the FrozenLake-v1 environment. The optimal state-value function and optimal policy were obtained, enabling the agent to identify the best possible actions to reach the goal state.
---

## Inference
1. Environment Initialization: The FrozenLake-v1 environment is created using the Gymnasium library.

2. State and Action Representation: The environment contains 16 states arranged in a 4×4 grid with four possible actions: Left, Down, Right, and Up.

3. Value Function Calculation: The Value Iteration algorithm calculates the optimal value of each state using iterative updates.

4. Optimal Policy Generation: The algorithm selects the best action for each state based on the calculated state values.

5. Output Visualization: The optimal state-value function, optimal policy, and FrozenLake grid layout are displayed to show the agent's path from the starting point to the goal.
```

