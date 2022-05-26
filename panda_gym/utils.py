import numpy as np
import math


def distance(a, b):
    assert a.shape == b.shape
    # print("scalars:", a, b)
    # d = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    # print("Distance calculated", d, "linalg:", np.linalg.norm(a - b, axis=-1))
    return np.linalg.norm(a - b, axis=-1)
