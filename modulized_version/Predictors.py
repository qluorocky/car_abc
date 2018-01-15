import numpy as np
import torch
import time
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
from torch import autograd


class Random_Predictor:
    def __init__(self, road_graph):
        self.road_graph = road_graph
    def _random_guess(self, node):
        n = len(self.road_graph.neighbors(node))
        return np.ones(n) / n
    def predict(self, path):
        cur = path[-1]
        # predict next node, and its distribution
        return self.road_graph.neighbors(cur), self._random_guess(cur)

class Markov_Predictor:
    def __init__(self, road_graph):
        self.road_graph = road_graph
        self.nn =     
    def predict(self, path):
        if not self.prediction:
            print("This Markvo_Predictor is not trained yet.")
            return None
        else:
            cur = path[-1]
            # predict next node, and its distribution
            return self.road_graph.neighbors(cur), self.prediction[cur]
    
    def train(self, data, batch_size = 50, learning_rate = 0.001, steps = 10000): # TODO
        # D_in, D_hidden, D_out = len(road_graph.nodes), 100, 4
        # model = torch.nn.Sequential(
        #     torch.nn.Linear(D_in, D_hidden),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(D_hidden, D_out),
        #     torch.nn.Softmax(dim = 1))
        # loss_fn = torch.nn.CrossEntropyLoss()
        # learning_rate = 0.002
        # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # for t in range(1,steps+1):
        #     x,y = random_training_set()
        #     y_pred = model(x)
        #     loss = loss_fn(y_pred, y)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()




class Naive_Markov_Predictor:
    # A markov predictor based on counting. (neural net free)
    def __init__(self, road_graph):
        self.road_graph = road_graph
        self.prediction = {}
    
    def predict(self, path):
        if not self.prediction:
            print("This Markvo_Predictor is not trained yet.")
            return None
        else:
            cur = path[-1]
            # predict next node, and its distribution
            return self.road_graph.neighbors(cur), self.prediction[cur]
    
    def train(self, data):
        # TODO
        pass


class RNN_Predictor:
    def __init__(self, road_graph):
        self.road_graph = road_graph
        self.parameters = {} # TODO
    
    def predict(self, path):
        if not self.parameters:
            print("This Markvo_Predictor is not trained yet.")
            return None
        else:
            # TODO
            pass
    
    def train(self, data):
        #TODO
        pass
