import numpy as np
import torch
import time
from sklearn.preprocessing import OneHotEncoder
from torch.autograd import Variable
from torch import nn
import matplotlib.pyplot as plt
from torch import autograd
from utils import log_loss

class Predictor:
    def __init__(self, road_graph):
        self.road_graph = road_graph
        self.model = None
        
    def predict(self, path):
        if not self.model:
            print("This predictor is not trained yet!")
            return None
        else:
            return self.road_graph.neighbors(path[-1]), self.model(path)
        
    def eval(self, test_data): # not fast, may need to improve speed (vectorize it!)
        loss = 0
        count = 0
        for path in test_data:
            p = [path[0]]
            for i in range(1, len(path)):
                nb, pred = self.predict(p)
                n = path[i]
                loss += log_loss(nb == n, pred)
                count += 1
                p.append(path[i])
                if count % 1000 == 0:
                    print("Processed {} road!".format(count))
        return loss/count

class Random_Predictor(Predictor):
    def _random_guess(self, node):
        n = len(self.road_graph.neighbors(node))
        return np.ones(n) / n
    def predict(self, path):
        cur = path[-1]
        return self.road_graph.neighbors(cur), self._random_guess(cur)

class Markov_Predictor(Predictor):
    def train(self, data):
        table = dict.fromkeys(self.road_graph.nodes)
        for k in table:
            table[k] = {n: 1 for n in self.road_graph.neighbors(k)}
        for path in data:
            for i in range(len(path) - 1):
                c,n = path[i], path[i+1]
                table[c][n] += 1
        for k in table:
            s = sum(list(table[k].values()))
            for n in table[k]:
                table[k][n] = table[k][n] / s
        def f(path):
            r = path[-1]
            return np.array(list(table[r].values()))
        self.model = f
    

class NN_Markov_Predictor(Predictor):
    def train(self, data):
        # TODO
        pass
    # def train(self, data, batch_size = 50, learning_rate = 0.001, steps = 10000):
    #     D_in, D_hidden, D_out = len(road_graph.nodes), 100, 4
    #     model = torch.nn.Sequential(
    #         torch.nn.Linear(D_in, D_hidden),
    #         torch.nn.ReLU(),
    #         torch.nn.Linear(D_hidden, D_out),
    #         torch.nn.Softmax(dim = 1))
    #     loss_fn = torch.nn.CrossEntropyLoss()
    #     learning_rate = 0.002
    #     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #     for t in range(1,steps+1):
    #         x,y = random_training_set()
    #         y_pred = model(x)
    #         loss = loss_fn(y_pred, y)
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()


class RNN_Predictor(Predictor):
    def train(self, data):
        #TODO
        pass


# class Markov_Predictor:
#     # A markov predictor based on counting. (neural net free)
#     def __init__(self, road_graph):
#         self.road_graph = road_graph
#         self.model = None    
#     def predict(self, path):
#         if not self.model:
#             print("This Markvo_Predictor is not trained yet.")
#             return None
#         else:
#             return self.road_graph.neighbors(path[-1]), self.model(path)
#     def train(self, data):
#         table = dict.fromkeys(self.road_graph.nodes)
#         for k in table:
#             table[k] = {n: 1 for n in self.road_graph.neighbors(k)}
#         for path in data:
#             for i in range(len(path) - 1):
#                 c,n = path[i], path[i+1]
#                 table[c][n] += 1
#         for k in table:
#             s = sum(list(table[k].values()))
#             for n in table[k]:
#                 table[k][n] = table[k][n] / s
#         def f(path):
#             r = path[-1]
#             return np.array(list(table[r].values()))
#         self.model = f

#     def eval(self, test_data):
#         # not fast, may need to improve speed (vectorize it!)
#         loss = 0
#         count = 0
#         for path in test_data:
#             p = [path[0]]
#             for i in range(1, len(path)):
#                 nb, pred = self.predict(p)
#                 n = path[i]
#                 loss += log_loss(nb == n, pred)
#                 count += 1
#                 p.append(path[i])
#                 if count % 100 == 0:
#                     print(count)
#         return loss/count
