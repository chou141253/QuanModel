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
from collections import namedtuple
import os

torch.cuda.empty_cache()

from float_model import Net
from quan_utils import *


FLOAT_MODEL_SAVE_PATH = './data_float/cifar10_float.pt'
QUAN_MODEL_SAVE_PATH = './data_quan/'


def train(model, device, loader, optimizer):
    
    model.train()
    for batch, (datas, labels) in enumerate(loader):
        datas, labels = datas.to(device), labels.to(device)
        optimizer.zero_grad()
        outs = model(datas)
        loss = F.cross_entropy(outs, labels)
        loss.backward()
        optimizer.step()
        
        if batch%100 == 0:
            print("    * Training-->batch:{}, loss:{:.4f}".format(batch, loss.item()))
        
        
def test(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch, (datas, labels) in enumerate(loader):
            datas, labels = datas.to(device), labels.to(device)
            outs = model(datas)
            test_loss += F.cross_entropy(outs, labels).sum().item()
            pred = outs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            
    test_loss /= len(loader.dataset)
    correct /= len(loader.dataset)

    print("    * Testing average loss: {:.4f}, average accuracy: {:.2f}%".format(test_loss, 100*correct))


def train_float():
    epoches = 60
    batch_size = 128
    data_size = 32
    device = 'cpu' # default device name
    if torch.cuda.is_available():
        device = 'cuda:0'
    model = Net(features_num=128, classes_num=10, dsize=32) # create model
    model = model.to(device)

    transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.RandomCrop(size=32, padding=4),
             transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703, 0.24349, 0.26159))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    optimizer = optim.Adam(model.parameters())

    for e in range(epoches):
        print("epoch: {}".format(e))
        print("start training...")
        train(model=model, device=device, loader=trainloader, optimizer=optimizer)
        print("training done.")
        print("start testing...")
        test(model=model, device=device, loader=testloader)
        print("testing done.\n")

    torch.save(model, FLOAT_MODEL_SAVE_PATH)
    print("save float model to ", FLOAT_MODEL_SAVE_PATH)

def merge_convbn():

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    model = torch.load(FLOAT_MODEL_SAVE_PATH).to(device)

    model_fuse3 = Net_fuse3(model=copy.deepcopy(model.to('cpu')), features_num=128, classes_num=10, dsize=32)
    model_fuse3 = model_fuse3.to(device)

    return model_fuse3

def adjust_learning_rate(optimizer, epoch, init_lr, adjustlr_rate):   
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    lr = init_lr * (0.4 ** (epoch // adjustlr_rate))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr


def save_stat(stats, save_path):

    for layer_name in stats:
        # read every layers information.
        for info_name in stats[layer_name]:
            if type(stats[layer_name][info_name]) == type(torch.tensor(0)):
                # transform tensor to float
                stats[layer_name][info_name] = float(stats[layer_name][info_name])

    with open(save_path, "w") as fjson:
        fjson.write(json.dumps(stats, indent=2))

def trainQuantAware(model, device, train_loader, optimizer, epoch, stats, act_quant=False, num_bits=4, records=None):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        output, stats, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight,\
        conv6weight, conv7weight, conv8weight, conv9weight, conv10weight, conv11weight, conv12weight,\
        classifier_weight = quantAwareTrainingForward(model, data, stats, num_bits=num_bits, act_quant=act_quant)

        model.conv1.weight.data = conv1weight
        model.conv2.weight.data = conv2weight
        model.conv3.weight.data = conv3weight
        model.conv4.weight.data = conv4weight
        model.conv5.weight.data = conv5weight
        model.conv6.weight.data = conv6weight
        model.conv7.weight.data = conv7weight
        model.conv8.weight.data = conv8weight
        model.conv9.weight.data = conv9weight
        model.conv10.weight.data = conv10weight
        model.conv11.weight.data = conv11weight
        model.conv12.weight.data = conv12weight
        model.classifier.weight.data = classifier_weight
    
        output = output.view(target.size(0), -1)
        
        #print("training prediction: {}".format(output))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        
        if batch_idx%50 == 0:
            records['train']['loss'].append(loss.item())

        if batch_idx%100 == 0:
            print('Train Epoch: {}   [{}/{} ({:3.0f}%)]  Loss: {:.6f}'.format(
                epoch, str(batch_idx*len(data)).zfill(5), len(train_loader.dataset),
                100.*batch_idx/len(train_loader), loss.item()))
    
    return stats, records

def testQuantAware(model, device, test_loader, stats, act_quant, num_bits=4):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            output, _, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight,\
            conv6weight, conv7weight, conv8weight, conv9weight, conv10weight, conv11weight, conv12weight,\
            classifier_weight = quantAwareTrainingForward(model, data, stats, num_bits=num_bits, act_quant=act_quant)

            model.conv1.weight.data = conv1weight
            model.conv2.weight.data = conv2weight
            model.conv3.weight.data = conv3weight
            model.conv4.weight.data = conv4weight
            model.conv5.weight.data = conv5weight
            model.conv6.weight.data = conv6weight
            model.conv7.weight.data = conv7weight
            model.conv8.weight.data = conv8weight
            model.conv9.weight.data = conv9weight
            model.conv10.weight.data = conv10weight
            model.conv11.weight.data = conv11weight
            model.conv12.weight.data = conv12weight
            model.classifier.weight.data = classifier_weight
            
            output = output.view(target.size(0), -1)
            #print("testingprediction: {}".format(output))
            
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return 100.*correct/len(test_loader.dataset), test_loss


def train_quan(model_fuse):
    
    batch_size = 128
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703, 0.24349, 0.26159))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


    model = copy.deepcopy(model_fuse)
    #optimizer = torch.optim.Adam(model.parameters())

    init_lr = 0.01
    optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

    num_bits = 8

    stats = {}

    init_lr = 0.001
    best_model_path = QUAN_MODEL_SAVE_PATH+"/cifar10_qun{}_best.pt".format(num_bits) # to save best accuracy model
    best_model_path_noq = QUAN_MODEL_SAVE_PATH+"/cifar10_qun{}_noq_best.pt".format(num_bits) # to save best accuracy model
    max_correct, max_correct_noq = -1, -1
    adjustlr_rate = 30
    epochs = 60

    records = {'train':{'acc':[], 'loss':[]}, 'test':{'acc':[], 'loss':[]}}

    print("initialize parameters successfully.")
    print("start running...\n")

    for epoch in range(10):
        
        lr_now = adjust_learning_rate(optimizer, epoch, init_lr, adjustlr_rate)
        
        if epoch<5:
            act_quant = False
        elif epoch<10:
            act_quant = True
        
        print("**Start epoch: {}, activation quant: {}, learning rate: {:.6f}. ".format(epoch+1, act_quant, lr_now))
        
        stats, records = trainQuantAware(model, device, trainloader, optimizer, epoch+1, stats, act_quant, num_bits=num_bits, records=records)

        corrects, test_loss = testQuantAware(model, device, testloader, stats, act_quant, num_bits=num_bits)
        records['test']['acc'].append(corrects) # record testing accuracy per epoch
        records['test']['loss'].append(test_loss) # record testing loss per epoch
        
        # to save best accuracy model(only under activation quantized mode)
        if act_quant and corrects>max_correct:
            print("  start saving best model (act_qun) ...")
            max_correct = corrects
            if os.path.isfile(best_model_path):
                # remove old file.
                os.remove(best_model_path)
            torch.save(model, best_model_path)
            print("  saving best model successfully.\n")
        
        if not act_quant and corrects>max_correct_noq:
            print("  start saving best model (noq) ...")
            max_correct_noq = corrects
            if os.path.isfile(best_model_path_noq):
                # remove old file.
                os.remove(best_model_path_noq)
            torch.save(model, best_model_path_noq)
            print("  saving best model successfully.\n")

    save_stat(stats, save_path=QUAN_MODEL_SAVE_PATH+"/stats.json")


if __name__ == "__main__":

    train_float()
    model_fuse3 = merge_convbn()
    train_quan(model_fuse3)