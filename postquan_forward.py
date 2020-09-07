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
from tqdm import trange, tqdm

torch.cuda.empty_cache()

from float_model import *
from quan_utils import *


def qforward():

    qnet = torch.load('./data_quan/cifar10_qun4_best.pt')
    stats = load_stats('./data_quan/stats_table4.json')

    #print(qnet)
    #print(stats)

    batch_size = 128

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703, 0.24349, 0.26159))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=4)
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # load model
    model = copy.deepcopy(qnet)

    num_bits = 4
    val_record = {'min':{}, 'max':{}}

    scale_factors_record = {}

    final_accuracys = {}

    corrects = 0
    total = len(testloader.dataset)
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for n_scale in range(1, 16):
            corrects = 0
            print("with n_scale : {}".format(n_scale))

            with tqdm(total=total, ascii=True) as tbar:
                for images, labels in testloader:

                    images, labels = images.cuda(), labels.cuda()
                    outputs, weights_record, bias_record, scale_factors_record = bit_conv_forward(images, 
                                                                                                   model, 
                                                                                                   num_bits,
                                                                                                   n_scale,
                                                                                                   val_record,
                                                                                                   scale_factors_record,
                                                                                                   stats)
                    model.conv1.weight.data = weights_record[0]
                    model.conv2.weight.data = weights_record[1]
                    model.conv3.weight.data = weights_record[2]
                    model.conv4.weight.data = weights_record[3]
                    model.conv5.weight.data = weights_record[4]
                    model.conv6.weight.data = weights_record[5]
                    model.conv7.weight.data = weights_record[6]
                    model.conv8.weight.data = weights_record[7]
                    model.conv9.weight.data = weights_record[8]
                    model.conv10.weight.data = weights_record[9]
                    model.conv11.weight.data = weights_record[10]
                    model.conv12.weight.data = weights_record[11]
                    model.classifier.weight.data = weights_record[12]
                    model.conv1.bias.data = bias_record[0]
                    model.conv2.bias.data = bias_record[1]
                    model.conv3.bias.data = bias_record[2]
                    model.conv4.bias.data = bias_record[3]
                    model.conv5.bias.data = bias_record[4]
                    model.conv6.bias.data = bias_record[5]
                    model.conv7.bias.data = bias_record[6]
                    model.conv8.bias.data = bias_record[7]
                    model.conv9.bias.data = bias_record[8]
                    model.conv10.bias.data = bias_record[9]
                    model.conv11.bias.data = bias_record[10]
                    model.conv12.bias.data = bias_record[11]
                    model.classifier.bias.data = bias_record[12]
                    _, predicted = torch.max(outputs, 1)
                    c = (predicted == labels).sum()

                    corrects += c

                    tbar.update(labels.size(0))

            acc = (100.*corrects/total)
            print("test accuracy: {}/{} = {}%\n".format(corrects, total, acc))
            final_accuracys[str(n_scale)] = acc

    return final_accuracys, scale_factors_record

if __name__ == "__main__":

    print("start forwarding...")
    final_accuracys, scale_factors_record = qforward()

    with open("fianl_accuracys.json", "w") as fjson:
        fjson.write(json.dumps(final_accuracys, indent=2))

    with open("scale_factors_record.json", "w") as fjson:
        fjson.write(json.dumps(scale_factors_record, indent=2))    
