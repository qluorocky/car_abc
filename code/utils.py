import numpy as np
import torch
from torch.autograd import Variable

def to_Variable(x, requires_grad = False):
    x = np.array(x).astype(np.float32)
    return Variable(torch.from_numpy(x), requires_grad = requires_grad)

def path_to_training_pair(path, graph):
    N = len(graph.nodes)
    n = len(path)
    inp = torch.from_numpy(np.float32(np.array([one_hot(p, N) for p in path[:-1]])))
    tar = torch.from_numpy(np.array([p for p in path[1:]]))
    return Variable(inp).contiguous(), Variable(tar).contiguous()

def training_set_old(path, graph):
    N = len(graph.nodes)
    n = len(path)
    inp = torch.from_numpy(np.float32(np.array([one_hot(p, N) for p in path[:-1]])))
    tar = [int(np.where(graph.neighbors(path[i]) == path[i+1])[0]) for i in range(n - 1)]
    tar = torch.from_numpy(np.array(tar))
    return Variable(inp).contiguous(), Variable(tar).contiguous()

def one_hot(x, N):
    l = np.zeros(N)
    l[x] = 1
    return l

def inv_one_hot(l):
    return np.argmax(l, axis=1)

def training_set(path, graph):
    N = len(graph.nodes)
    n = len(path)
    inp = torch.from_numpy(np.float32(np.array([one_hot(p, N) for p in path[:-1]])))
    tar = torch.from_numpy(np.array([p for p in path[1:]]))
    return Variable(inp).contiguous(), Variable(tar).contiguous()

def log_loss(p, p_):
    # e.g. p = [T,F,T], p_ = [1/3,1/3,1/3]
    return -np.log(p_[p][0])
