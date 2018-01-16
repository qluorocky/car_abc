import numpy as np

def one_hot(x, N):
    l = np.zeros(N)
    l[x] = 1
    return l

def inv_one_hot(l):
    return np.argmax(l)

