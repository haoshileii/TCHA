import numpy as np
import pandas as pd
from collections import defaultdict
def fastdtw(x, y, radius=1, dist=lambda a, b: np.sum(np.abs(a - b))):
    min_time_size = radius + 2
    if len(x) < min_time_size or len(y) < min_time_size:
        return dtw(x, y, dist=dist)
    x_shrinked = __reduce_by_half(x)
    y_shrinked = __reduce_by_half(y)
    distance, path = fastdtw(x_shrinked, y_shrinked, radius=radius, dist=dist)
    window = __expand_window(path, len(x), len(y), radius)
    return dtw(x, y, window, dist=dist)
def dtw(x, y, window=None, dist=lambda a, b: np.sum(np.abs(a - b))):
    len_x, len_y = len(x), len(y)
    if window is None:
        window = [(i, j) for i in range(len_x) for j in range(len_y)]
    window = ((i + 1, j + 1) for i, j in window)

    D = defaultdict(lambda: (float('inf'),))
    D[0, 0] = (0, 0, 0)
    for i, j in window:
        dt = dist(x[ i -1], y[ j -1])
        D[i, j] = min((D[ i -1, j][0] +dt, i- 1 , j  ),
                      (D[i, j- 1][0] + dt, i, j - 1),
                      (D[i - 1, j - 1][0] + dt, i - 1, j - 1), key=lambda a: a[0])
    path = []

    i, j = len_x, len_y
    while not (i == j == 0):
        path.append((i - 1, j - 1))
        i, j = D[i, j][1], D[i, j][2]
    path.reverse()
    return (D[len_x, len_y][0], path)
def __expand_window(path, len_x, len_y, radius):
    path_ = set(path)
    for i, j in path:
        for a, b in ((i + a, j + b)
                     for a in range(-radius, radius + 1)
                     for b in range(-radius, radius + 1)):
            path_.add((a, b))
    window_ = set()
    for i, j in path_:
        for a, b in ((i * 2, j * 2), (i * 2, j * 2 + 1),
                     (i * 2 + 1, j * 2), (i * 2 + 1, j * 2 + 1)):
            window_.add((a, b))

    window = []
    start_j = 0
    for i in range(0, len_x):
        new_start_j = None
        for j in range(start_j, len_y):
            if (i, j) in window_:
                window.append((i, j))
                if new_start_j is None:
                    new_start_j = j
            elif new_start_j is not None:
                break
        start_j = new_start_j
    return window
def __reduce_by_half(x):
    """
    input list, make each two element together by half of sum of them
    :param x:
    :return:
    """
    x_reduce = []
    lens = len(x)
    for i in range(0, lens, 2):
        if (i + 1) >= lens:
            half = x[i]
        else:
            half = (x[i] + x[i + 1]) / 2
        x_reduce.append(half)
    return x_reduce
