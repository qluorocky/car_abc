# https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.gru = nn.GRU(input_size, hidden_size, n_layers)
        self.tran = nn.Linear(hidden_size, output_size)
        
        hidden0 = torch.zeros(n_layers, 1, hidden_size)
#         if use_cuda:
#             hidden0 = hidden0.cuda()
#         else:
#             hidden0 = hidden0
        self.hidden0 = nn.Parameter(hidden0, requires_grad=True)
    
    def forward(self, inp, hidden):
        output, hidden = self.gru(inp.view(1, 1, input_size), hidden)
        output = self.tran(output) 
        return output, hidden

#     def init_hidden(self):
#         return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    def init_hidden(self):
        return self.hidden0

