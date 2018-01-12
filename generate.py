import torch

from helpers import *
from model import *

def generate(decoder, start, predict_len=100):
    """
    first consider start is a [time, lat, lng] triple, may generate it into a path.
    Think generate a idle path condition on its picking path
    
    Think how to add randomness into the simulated path! (temperature)
    """ 
    hidden = decoder.init_hidden()
    prime_input = to_var(start)
    predicted = [start]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        # Add predicted triple into "predicted" and use it as next input
        predicted.append(output)
        inp = output
    return predicted
