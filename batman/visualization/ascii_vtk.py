import numpy as np
from itertools import cycle

n_dim = 3
n_stack = 6

# 0 3 6 9
base = np.array([i * n_dim for i in range(n_stack)])
base = np.array([base + i for i in range(n_dim * 2)]).T

first_cycle = [i for i in range(n_dim)]
second_cycle = [i for i in range(n_dim, n_dim * 2)]

# 01 12 23 30
first_stack = []
second_stack = []
for i in range(n_dim):
    first_stack.append([first_cycle[i], first_cycle[i] + 1])
    second_stack.append([second_cycle[i], second_cycle[i] + 1])

first_stack[-1][1] = 0
second_stack[-1][1] = n_dim

first_cells = np.array(list(zip(first_stack, second_stack))).reshape(-1, 4)

out = [base[:, first_cells[i]] for i in range(n_dim)]
connectivity = np.array(list(zip(*out))).reshape(-1, 4)

print(connectivity)
