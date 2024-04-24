import numpy as np

def maximum_path_each(path, value, t_y, t_x, max_neg_val=-1e9):
    index = t_x - 1

    for y in range(t_y):
        for x in range(max(0, t_x + y - t_y), min(t_x, y + 1)):
            if x == y:
                v_cur = max_neg_val
            else:
                v_cur = value[y-1, x] if y > 0 else max_neg_val

            if x == 0:
                v_prev = 0. if y == 0 else max_neg_val
            else:
                v_prev = value[y-1, x-1]

            value[y, x] += max(v_prev, v_cur)

    for y in range(t_y - 1, -1, -1):
        path[y, index] = 1
        if index != 0 and (index == y or value[y-1, index] < value[y-1, index-1]):
            index = index - 1

def maximum_path_c(paths, values, t_ys, t_xs):
    b = paths.shape[0]
    for i in range(b):
        maximum_path_each(paths[i], values[i], t_ys[i], t_xs[i])

# Example usage:
# paths = np.zeros((b, t_y, t_x), dtype=int)
# values = np.zeros((b, t_y, t_x), dtype=float)
# t_ys = np.array([3, 4, 2])
# t_xs = np.array([4, 5, 3])
# maximum_path(paths, values, t_ys, t_xs)
