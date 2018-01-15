def one_hot(x, N = 24):
    l = np.zeros(24)
    l[x] = 1
    return l

def inv_one_hot(l, N = 24):
    return np.argmax(l)

