from tf_models.base.train import BaseTrain
from tqdm import tqdm
import numpy as np

class Trainer(BaseTrain):
    def __init__(self, sess, model, data, options):
        super(Trainer, self).__init__(sess, model, data, options)
