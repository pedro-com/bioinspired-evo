import numpy as np
import matplotlib.pyplot as plt

def fit_queen(queen_positions: np.ndarray):
    N = queen_positions.shape[0]
    queen_pos = np.array([(line, column) for line, column in enumerate(queen_positions)])
    penalization = 0
    for pos in queen_pos:
        change = queen_pos - pos
        penalization += (change[:, 0] == change[:, 1]).sum()
        penalization += (change[:, 0] == -change[:, 1]).sum()
    return penalization - 2*N

def visualize_queens_board(queen_positions: np.ndarray, figsize=(12, 6), with_diagonals: bool=True):

    N = len(queen_positions)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw the chessboard
    for row in range(N):
        for col in range(N):
            # color = 'lightgray' if (row + col) % 2 == 0 else 'darkgray'
            color = 'white' if (row + col) % 2 == 0 else 'black'
            ax.add_patch(plt.Rectangle((col, N - row - 1), 1, 1, color=color, ec="black"))
    
    cmap = plt.get_cmap("hsv")
    # Draw the queens
    for col, row in enumerate(queen_positions):
        queen_color = cmap(col / N) if with_diagonals else 'red'
        ax.add_patch(plt.Circle((col + 0.5, N - row - 0.5), 0.3, color=queen_color, ec=queen_color, lw=2))
        if not with_diagonals:
            continue
        # Draw diagonal lines
        for offset in range(1, N):
            # Positive slope diagonal (\ direction)
            if 0 <= col + offset < N and 0 <= row + offset < N:
                ax.plot([col + 0.5, col + offset + 0.5], [N - row - 0.5, N - (row + offset) - 0.5], color=queen_color, linestyle='-', lw=0.5)
            if 0 <= col - offset < N and 0 <= row - offset < N:
                ax.plot([col + 0.5, col - offset + 0.5], [N - row - 0.5, N - (row - offset) - 0.5], color=queen_color, linestyle='-', lw=0.5)

            # Negative slope diagonal (/ direction)
            if 0 <= col + offset < N and 0 <= row - offset < N:
                ax.plot([col + 0.5, col + offset + 0.5], [N - row - 0.5, N - (row - offset) - 0.5], color=queen_color, linestyle='-', lw=0.5)
            if 0 <= col - offset < N and 0 <= row + offset < N:
                ax.plot([col + 0.5, col - offset + 0.5], [N - row - 0.5, N - (row + offset) - 0.5], color=queen_color, linestyle='-', lw=0.5)
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_aspect('equal')
    ax.axis('off')  # Hide the axis

    ax.set_title(f'{N}-Queens Solution', fontsize=16)


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