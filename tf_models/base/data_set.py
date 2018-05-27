import numpy as np

class BaseDataSet:
    def __init__(self, options):
        self.options = options

    def next_train_batch(self, batch_size):
        raise NotImplementedError

    def test_set(self):
        '''Returns test set.

        Returns:
            x
            y 
        '''
        raise NotImplementedError
        
