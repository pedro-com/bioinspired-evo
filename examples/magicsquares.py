import numpy as np

def fit_square(square: np.ndarray, max_repetitions:float=1.):
    N = square.shape[0]
    diag_v = np.arange(N)
    columns = np.sum(square, axis=0)
    lines = np.sum(square, axis=1)
    diags = np.sum([square[diag_v, diag_v], square[np.flip(diag_v), diag_v]], axis=1)
    median = np.median(np.concatenate([columns, lines, diags]))
    penalization = np.sum(np.abs(columns - median)) + np.sum(np.abs(lines - median)) + np.sum(np.abs(diags - median))
    _, counts = np.unique(square, return_counts=True)
    if np.max(counts) > max_repetitions*N**2:
        penalization *= 2
    return penalization

def to_msquare(cromosome: np.ndarray, ms_size:int):
    return cromosome.reshape(ms_size, ms_size)

