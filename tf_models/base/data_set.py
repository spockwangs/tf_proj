import numpy as np

class BaseDataSet:
    def __init__(self, options):
        self.options = options

    def next_batch(self, batch_size):
        raise NotImplementedError
