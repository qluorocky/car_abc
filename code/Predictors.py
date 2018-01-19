import numpy as np
import time
import torch
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from utils import log_loss
from Neural_Network import *


class Predictor:
    def __init__(self, graph):
        self.graph = graph
        self.model = None
        
    def predict(self, path):
        if not self.model:
            print("This predictor is not trained yet!")
            return None
        else:
            return self.graph.neighbors(path[-1]), self.model(path)   
    def evaluate(self, test_data): # not fast, may need to improve speed (vectorize it!)
        if not self.model:
            print("This predictor is not trained yet!")
            return None
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
        n = len(self.graph.neighbors(node))
        return np.ones(n) / n
    def predict(self, path):
        cur = path[-1]
        return self.graph.neighbors(cur), self._random_guess(cur)

class Markov_Predictor(Predictor):
    def train(self, data):
        table = dict.fromkeys(self.graph.nodes)
        for k in table:
            table[k] = {n: 1 for n in self.graph.neighbors(k)}
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
        pass
    # def train(self, data, batch_size = 50, learning_rate = 0.001, steps = 10000):
    #     D_in, D_hidden, D_out = len(graph.nodes), 100, 4
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
    def __init__(self, graph, im_D = 50, h_D = 100):
        super().__init__(graph)
        self.W = Variable(torch.randn(graph.num_nodes, im_D).type(torch.FloatTensor), requires_grad=True) # we first use random embeding (may use road2vec later)
        self.rnn = RNN(im_D, h_D, im_D)
        self.model = None 

    def _path_loss(self, path):
        W = self.W
        h = self.rnn.init_hidden()
        output_list = []
        for i in path:
            o, h = self.rnn(W[i], h)
            output_list.append(o)
        loss_list = []
        for i in range(len(path) - 1):
            x = path[i]
            y = path[i+1]
            nb = self.graph.neighbors(x)
            W_list = [W[j].view(-1,1) for j in nb]
            Ws = torch.cat(W_list, dim = 1)
            W_next = output_list[i]
            W_next_rep = W_next.view(-1,1).repeat(1,Ws.size()[1])
            logit = torch.sum(W_next_rep*Ws, 0)
            prob = F.softmax(logit, dim = 0)
            loss_list.append(-torch.log(prob[np.where(nb == y)]))
        return torch.sum(torch.cat(loss_list))

    def _get_prob(path):
        W = self.W
        h = self.rnn.init_hidden()
        for i in path:
            o, h =self.rnn(W[i], h)
        x = path[-1]
        nb = self.graph.neighbors(x)
        W_list = [W[j].view(-1,1) for j in nb]
        Ws = torch.cat(W_list, dim = 1)
        W_next = o
        W_next_rep = W_next.view(-1,1).repeat(1,Ws.size()[1])
        logit = torch.sum(W_next_rep*Ws, 0)
        prob = F.softmax(logit, dim = 0)
        return prob
    
    def train(self, data, lr = 0.001, batch_size = 50):
        
        optimizer = torch.optim.Adam(self.rnn.parameters(), lr=0.001) # remove [W] for a fixed random embeding
        loss_list = []
        s = 0
        for k in range(len(data)):
            path = data[k]
            loss = self._path_loss(path)
            loss_list.append(loss)
            s += len(path)
            if k%batch_size == 0:
                batch_loss = torch.sum(torch.cat(loss_list)) / s
                s = 0
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                loss_list = []
                print(batch_loss.data.numpy())
        self.model = self._get_prob


# class Markov_Predictor:
#     # A markov predictor based on counting. (neural net free)
#     def __init__(self, graph):
#         self.graph = graph
#         self.model = None    
#     def predict(self, path):
#         if not self.model:
#             print("This Markvo_Predictor is not trained yet.")
#             return None
#         else:
#             return self.graph.neighbors(path[-1]), self.model(path)
#     def train(self, data):
#         table = dict.fromkeys(self.graph.nodes)
#         for k in table:
#             table[k] = {n: 1 for n in self.graph.neighbors(k)}
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
