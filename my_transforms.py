import torch
import torch.nn as nn

class Float2Uint(object):
    def __call__(self, tensors):
        tensors = 255.*tensors
        tensors.round_()
        return tensors
    
    def __repr__(self):
        return self.__class__.__name__ + '()'