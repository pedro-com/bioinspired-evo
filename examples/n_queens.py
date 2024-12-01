import numpy as np

def fit_queen(queen_positions: np.ndarray):
    N = queen_positions.shape[0]
    queen_pos = np.array([(line, column) for line, column in enumerate(queen_positions)])
    penalization = 0
    for pos in queen_pos:
        change = queen_pos - pos
        penalization += (change[:, 0] == change[:, 1]).sum()
        penalization += (change[:, 0] == -change[:, 1]).sum()
    return penalization - 2*N

def print_queens(queen_positions: np.ndarray):
    N = queen_positions.shape[0]
    for pos in queen_positions:
        queen_line = " | ".join("Q" if k == pos else "+" for k in range(N))
        print(f"|{queen_line}|")

'''
0 Q 0 0 0
0 0 0 Q 0
Q 0 0 0 0
0 0 Q 0 0
0 0 0 0 Q

for line, column in enumerate(queen_positions):
    queen_mask = np.zeros((N, N), dtype=np.uint16)
    queen_mask[line, :] += 1
    queen_mask[:, column] += 1
    if line == column:
        diag_p = np.array(tuple(zip(range_N, range_N)))
        diag_n = np.array(tuple(zip(range_N, np.flip(range_N))))
    elif line < column:
        diag_p = np.array([(k, column - line + k) for k in range_N if 0 <= column - line + k < N])
        diag_n = np.array([(N - 1 - k, column - line + k) for k in range_N if 0 <= column - line + k < N])
    else:
        diag_p = np.array([(k, line - column + k) for k in range_N if 0 <= line - column + k < N])
        diag_n = np.array([(N - 1 - k, line - column + k) for k in range_N if 0 <= line - column + k < N])
    for pos in np.concatenate([diag_p, diag_n]):
        queen_mask[pos[0], pos[1]] += 1
    queen_mask[line, column] -= 3
'''