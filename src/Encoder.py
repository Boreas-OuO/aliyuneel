import numpy as np
import einops

class Encoder():
    def __init__(self, config):
        self.I_size = config['window_len']
        self.O_size = config['O_size']
        self.flip = config['flip']
        self.d