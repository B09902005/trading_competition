import os
import torch
from models import iTransformer

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer,
        }

        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if torch.cuda.is_available():
            print("Using GPU")
            return torch.device('cuda')
        else:
            print("No GPU available")
            #exit()
            return torch.device('cpu')

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
