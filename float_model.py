import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt
import json

torch.cuda.empty_cache()

from my_transforms import Float2Uint

class Net(nn.Module):
    
    def __init__(self, features_num=128, classes_num=10, dsize=32):
       
        super(Net, self).__init__()
        
        self.classes_num = classes_num
        
        self.conv1 = nn.Conv2d(3, features_num, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv2 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv2 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv3 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv4 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        
        self.conv5 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv6 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv7 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv8 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn8 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        
        self.conv9 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn9 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv10 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn10 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv11 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn11 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        self.conv12 = nn.Conv2d(features_num, features_num, 3, 1, 1, bias=False)
        self.bn12 = nn.BatchNorm2d(features_num, eps=0.00001, momentum=0.1)
        
        self.classifier = nn.Conv2d(features_num, classes_num, dsize//8, 1)
        #self.bn_classifier = nn.BatchNorm2d(classes_num, eps=0.001, momentum=0.1)
        
    def forward(self, x):
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.max_pool2d(x, 2, 2)
        
        x = self.classifier(x)
        #x = self.bn_classifier(x)
        #x = F.relu(x)
        
        x = x.view(-1, self.classes_num)
        
        return F.log_softmax(x, dim=1)

def fuse(conv, bn):
    w = conv.weight
    mean = bn.running_mean
    var_sqrt = torch.sqrt(bn.running_var + bn.eps)

    beta = bn.weight
    gamma = bn.bias

    if conv.bias is not None:
        b = conv.bias
    else:
        b = mean.new_zeros(mean.shape)

    w = w * (beta / var_sqrt).reshape([conv.out_channels, 1, 1, 1])
    b = (b - mean)/var_sqrt * beta + gamma
    fused_conv = nn.Conv2d(conv.in_channels,
                         conv.out_channels,
                         conv.kernel_size,
                         conv.stride,
                         conv.padding,
                         bias=True)
    
    fused_conv.weight = nn.Parameter(w)
    fused_conv.bias = nn.Parameter(b)
    return fused_conv


class Net_fuse3(nn.Module):
    
    def __init__(self, model, features_num, classes_num, dsize):
        
        super(Net_fuse3, self).__init__()
        
        self.classes_num = classes_num
        
        self.conv1 = fuse(model.conv1, model.bn1)
        self.conv2 = fuse(model.conv2, model.bn2)
        self.conv3 = fuse(model.conv3, model.bn3)
        self.conv4 = fuse(model.conv4, model.bn4)
        self.conv5 = fuse(model.conv5, model.bn5)
        self.conv6 = fuse(model.conv6, model.bn6)
        self.conv7 = fuse(model.conv7, model.bn7)
        self.conv8 = fuse(model.conv8, model.bn8)
        self.conv9 = fuse(model.conv9, model.bn9)
        self.conv10 = fuse(model.conv10, model.bn10)
        self.conv11 = fuse(model.conv11, model.bn11)
        self.conv12 = fuse(model.conv12, model.bn12)
        
        #self.classifier = fuse(model.classifier, model.bn_classifier)
        self.classifier = copy.deepcopy(model.classifier)
        
    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = F.relu(self.conv8(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = F.relu(self.conv9(x))
        x = F.relu(self.conv10(x))
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = F.max_pool2d(x, 2, 2)
        
        x = self.classifier(x)
        
        x = x.view(-1, self.classes_num)
        
        return F.log_softmax(x, dim=1)