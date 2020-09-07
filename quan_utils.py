from collections import namedtuple
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

import postquan_model as pqmodel


QTensor = namedtuple('QTensor', ['tensor', 'scale', 'zero_point'])


def updateStats(x, stats, key):
    max_val, _ = torch.max(x, dim=1)
    min_val, _ = torch.min(x, dim=1)

    # add ema calculation

    if key not in stats:
        stats[key] = {"max": max_val.sum(), "min": min_val.sum(), "total": 1}
    else:
        stats[key]['max'] += max_val.sum().item()
        stats[key]['min'] += min_val.sum().item()
        stats[key]['total'] += 1

    weighting = 2.0 / (stats[key]['total']) + 1

    if 'ema_min' in stats[key]:
        stats[key]['ema_min'] = weighting*(min_val.mean().item()) + (1- weighting) * stats[key]['ema_min']
    else:
        stats[key]['ema_min'] = weighting*(min_val.mean().item())

    if 'ema_max' in stats[key]:
        stats[key]['ema_max'] = weighting*(max_val.mean().item()) + (1- weighting) * stats[key]['ema_max']
    else: 
        stats[key]['ema_max'] = weighting*(max_val.mean().item())

    stats[key]['min_val'] = stats[key]['min']/ stats[key]['total']
    stats[key]['max_val'] = stats[key]['max']/ stats[key]['total']

    return stats

def calcScaleZeroPoint(min_val, max_val,num_bits=8):
    # Calc Scale and zero point of next 
    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0:
        print("[WARN] scale is zero.")
        return 1, qmin

    initial_zero_point = qmin - min_val / scale
  
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = int(zero_point)

    return scale, zero_point


def quantize_tensor(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()

    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().byte()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point)

def quantize_tensor_w(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()

    qmin = -2.**(num_bits - 1.) + 1.
    qmax = 2.**(num_bits - 1.) - 1.
    
    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    q_x = zero_point + x / scale
    q_x = q_x - 2**(num_bits - 1)
    q_x.clamp_(qmin, qmax).round_()
    #     q_x = q_x + 2**(num_bits - 1)
    q_x = q_x.round().char()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point-2**(num_bits - 1))

def dequantize_tensor(q_x):
    return q_x.scale * (q_x.tensor.float() - q_x.zero_point)

def get_indexs(x, index):

    for i in range(len(x.size())):
        total = 1
        for j in range(i+1, len(x.size())):
            total *= x.size(j)
        indexs.append(math.floor(index/total))
        index -= indexs[-1]*total
    
    return indexs

def calcScaleZeroPoint_w(min_val, max_val,num_bits=8):
    # Calc Scale and zero point of next 
    qmin = 0.
    qmax = 2.**num_bits - 2.

    scale = (max_val - min_val) / (qmax - qmin)
    if scale == 0:
        print("[WARN] scale is zero.")
        return 1, qmin

    initial_zero_point = qmin - min_val / scale
    #     print("qmin, min_val, scale, initial_zero_point : {}, {}, {}, {}".format(qmin, min_val, scale, initial_zero_point))
  
    zero_point = 0
    if initial_zero_point < qmin:
        zero_point = qmin
    elif initial_zero_point > qmax:
        zero_point = qmax
    else:
        zero_point = initial_zero_point

    zero_point = zero_point.round()

    return scale, zero_point

def quantize_tensor2(x, num_bits=8, min_val=None, max_val=None):
    
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()
        
    if abs(min_val)>max_val:
        index = int(x.argmax().detach())
        indexs = get_indexs(x, index)
        x[indexs[0]][indexs[1]][indexs[2]] = -1*min_val
        max_val = -1*min_val
    else:
        index = int(x.argmin().detach())
        indexs = get_indexs(x, index)
        x[indexs[0]][indexs[1]][indexs[2]] = -1*max_val
        min_val = -1*max_val
        
    qmin = -2.**(num_bits - 1.) + 1.
    qmax = 2.**(num_bits - 1.) - 1.
    
    scale, zero_point = calcScaleZeroPoint_w(min_val, max_val, num_bits)
    #     print ("scale : {} , zero_point : {}".format(scale, zero_point))
    q_x = zero_point + x / scale
    q_x = q_x - (2**(num_bits - 1)-1)
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round().char()
    
    return QTensor(tensor=q_x, scale=scale, zero_point=zero_point-(2**(num_bits - 1)-1))

class FakeQuantOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
        x = quantize_tensor(x,num_bits=num_bits, min_val=min_val, max_val=max_val)
        x = dequantize_tensor(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None
    

class FakeQuantOp_w(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_bits=8, min_val=None, max_val=None):
    #         print ("Before quantize :\n {}".format(x))
        x = quantize_tensor2(x, num_bits=num_bits, min_val=min_val, max_val=max_val)
    #         print ("After quantize :\n {}".format(x))
        x = dequantize_tensor(x)
    #         print ("After dequantize :\n {}".format(x))
        return x

    @staticmethod
    def backward(ctx, grad_output):
        # straight through estimator
        return grad_output, None, None, None

def quantAwareTrainingForward(model, x, stats, num_bits=8, act_quant=False):
    
    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'inputs')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['inputs']['ema_min'], stats['inputs']['ema_max'])
    
    """ 1 layer """
    conv1weight = model.conv1.weight.data
    model.conv1.weight.data = FakeQuantOp_w.apply(model.conv1.weight.data, num_bits) # quantized weight
    x = F.relu(model.conv1(x)) # use quantized weight to do convolution

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv1')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])
            
    """ 2 layer """
    conv2weight = model.conv2.weight.data
    model.conv2.weight.data = FakeQuantOp_w.apply(model.conv2.weight.data, num_bits)
    x = F.relu(model.conv2(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv2')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv2']['ema_min'], stats['conv2']['ema_max'])
            
    """ 3 layer """
    conv3weight = model.conv3.weight.data
    model.conv3.weight.data = FakeQuantOp_w.apply(model.conv3.weight.data, num_bits)
    x = F.relu(model.conv3(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv3')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv3']['ema_min'], stats['conv3']['ema_max']) 
        
    """ 4 layer """
    conv4weight = model.conv4.weight.data
    model.conv4.weight.data = FakeQuantOp_w.apply(model.conv4.weight.data, num_bits)
    x = F.relu(model.conv4(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv4')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv4']['ema_min'], stats['conv4']['ema_max']) 

    x = F.max_pool2d(x, 2, 2)

    """ 5 layer """
    conv5weight = model.conv5.weight.data
    model.conv5.weight.data = FakeQuantOp_w.apply(model.conv5.weight.data, num_bits)
    x = F.relu(model.conv5(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv5')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv5']['ema_min'], stats['conv5']['ema_max'])
            
    """ 6 layer """
    conv6weight = model.conv6.weight.data
    model.conv6.weight.data = FakeQuantOp_w.apply(model.conv6.weight.data, num_bits)
    x = F.relu(model.conv6(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv6')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv6']['ema_min'], stats['conv6']['ema_max'])
            
    """ 7 layer """
    conv7weight = model.conv7.weight.data
    model.conv7.weight.data = FakeQuantOp_w.apply(model.conv7.weight.data, num_bits)
    x = F.relu(model.conv7(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv7')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv7']['ema_min'], stats['conv7']['ema_max']) 
        
    """ 8 layer """
    conv8weight = model.conv8.weight.data
    model.conv8.weight.data = FakeQuantOp_w.apply(model.conv8.weight.data, num_bits)
    x = F.relu(model.conv8(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv8')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv8']['ema_min'], stats['conv8']['ema_max']) 

    x = F.max_pool2d(x, 2, 2)
    
    """ 9 layer """
    conv9weight = model.conv9.weight.data
    model.conv9.weight.data = FakeQuantOp_w.apply(model.conv9.weight.data, num_bits)
    x = F.relu(model.conv9(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv9')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv9']['ema_min'], stats['conv9']['ema_max'])
            
    """ 10 layer """
    conv10weight = model.conv10.weight.data
    model.conv10.weight.data = FakeQuantOp_w.apply(model.conv10.weight.data, num_bits)
    x = F.relu(model.conv10(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv10')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv10']['ema_min'], stats['conv10']['ema_max'])
            
    """ 11 layer """
    conv11weight = model.conv11.weight.data
    model.conv11.weight.data = FakeQuantOp_w.apply(model.conv11.weight.data, num_bits)
    x = F.relu(model.conv11(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv11')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv11']['ema_min'], stats['conv11']['ema_max']) 
        
    """ 12 layer """
    conv12weight = model.conv12.weight.data
    model.conv12.weight.data = FakeQuantOp_w.apply(model.conv12.weight.data, num_bits)
    x = F.relu(model.conv12(x))

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'conv12')

    if act_quant:
        x = FakeQuantOp.apply(x, num_bits, stats['conv12']['ema_min'], stats['conv12']['ema_max']) 

    x = F.max_pool2d(x, 2, 2)
    
    
    """ classifier layer """
    classifier_weight = model.classifier.weight.data
    model.classifier.weight.data = FakeQuantOp_w.apply(model.classifier.weight.data, num_bits)
    x = model.classifier(x)

    with torch.no_grad():
        stats = updateStats(x.clone().view(x.shape[0], -1), stats, 'classifier')
    
    return F.log_softmax(x, dim=1), stats, conv1weight, conv2weight, conv3weight, conv4weight, conv5weight,\
           conv6weight, conv7weight, conv8weight, conv9weight, conv10weight, conv11weight, conv12weight, classifier_weight


def fit_biases(x, s1, s2, mode=1):
    if mode == 0:
        x = x*0
    else:
        x = x/(s1*s2)
        x.round()
    return x

# same as quantized_tensor function but not use QTensor as datatype
def fit_weights(x, num_bits=4, min_val=None, max_val=None):
    
    if not min_val and not max_val:
        min_val, max_val = x.min(), x.max()

    if abs(min_val)>max_val:
        index = int(x.argmax().detach())
        indexs = get_indexs(x, index)
        x[indexs[0]][indexs[1]][indexs[2]] = -1*min_val
        max_val = -1*min_val
    else:
        index = int(x.argmin().detach())
        indexs = get_indexs(x, index)
        x[indexs[0]][indexs[1]][indexs[2]] = -1*max_val
        min_val = -1*max_val
        
    qmin = -2.**(num_bits - 1.) + 1.
    qmax = 2.**(num_bits - 1.) - 1.
    
    scale, zero_point = calcScaleZeroPoint_w(min_val, max_val, num_bits)
#     print("zero_point by (w)function", zero_point)
    q_x = zero_point + x / scale
    q_x = q_x - (2**(num_bits - 1)-1)
    zero_point = zero_point - (2**(num_bits - 1)-1)
#     print("zero_point after bit-shift", zero_point)
    q_x.clamp_(qmin, qmax).round_()
    q_x = q_x.round()
    
    return q_x, min_val, max_val


def bit_shift(x, keep_bits_num):
    # max_val = x.max()
    # max_bits = torch.ceil(torch.log2(max_val))
    # shift = max_bits - keep_bits_num # calculate shift bits number
    shift = 10 - keep_bits_num
    shift = max(0, shift)
    x = torch.floor(x/(2**(shift)))
    return x, shift

def ScalebyM(x, M, shift):
    # x = M*x
    x = M*x*(2**(shift))
    x.clamp_(0, 7).round_()
    return x
    
def get_M(stats, s2, this_layer, prvs_layer, num_bits):
    
    # min_val = stats[layer]['ema_min'] # get this layers 
    # max_val = stats[layer]['ema_max']
    
    """
        M = S1 * S2 / S3
        S1 = prvs layer's activation
        S2 = this layer's weight
        S3 = this layer's activation
    """
    
    s1 = get_scale(stats[prvs_layer]['ema_min'], stats[prvs_layer]['ema_max'], num_bits)
    s3 = get_scale(stats[this_layer]['ema_min'], stats[this_layer]['ema_max'], num_bits)
    
    M = s1*s2/s3
    return s1, s3, M

def get_minmax(val_record, min_val, max_val, model_layer):
    if model_layer not in val_record['min'] and model_layer not in val_record['max']:
        val_record['min'][model_layer] = min_val
        val_record['max'][model_layer] = max_val
    else:
        min_val = val_record['min'][model_layer]
        max_val = val_record['max'][model_layer]
        
    return min_val, max_val

def get_scale(min_val, max_val, num_bits):
    scale = (max_val - min_val) / (2**(num_bits)-1)
    return scale

def get_scale_w(min_val, max_val, num_bits):
    scale = (max_val - min_val) / (2**(num_bits)-2)
    return scale

def load_stats(path):
    datas = None
    with open(path, "r") as fjson:
        datas = json.load(fjson)
    stats = {}
    for name in datas:
        tmp_data = datas[name] # load layer data
        stats[name] = {'ema_min':tmp_data['ema_min'],
                       'ema_max':tmp_data['ema_max'],
                      }
    return stats

def record_scale123(scale_factors_record, this_layer, s1, s2, s3):
    if this_layer not in scale_factors_record:
        scale_factors_record[this_layer] = {"s1":s1, "s2":s2.item(), "s3":s3}
    return scale_factors_record

def record_maxvalue(scale_factors_record, this_layer, max_val):
    if 'max_conv_out' not in scale_factors_record[this_layer]:
        scale_factors_record[this_layer]['max_conv_out'] = []
    scale_factors_record[this_layer]['max_conv_out'].append(max_val)
    return scale_factors_record

def bit_conv_layer(x, model_layer, stats, num_bits, n_scale, this_layer, prvs_layer, val_record, scale_factors_record):
    
    
    model_layer.weight.data, min_val, max_val = fit_weights(model_layer.weight.data, num_bits) # fit weight to 0~255(with 8bits)
    min_val, max_val = get_minmax(val_record, min_val, max_val, this_layer) # get this layer's max and min val (weights)
    s2 = get_scale_w(min_val, max_val, num_bits) # get scale value
    s1, s3, M = get_M(stats, s2, this_layer, prvs_layer, num_bits)
    scale_factors_record = record_scale123(scale_factors_record, this_layer, s1, s2, s3)
    
    model_layer.bias.data = fit_biases(model_layer.bias.data, s1, s2)
    
    # original_w_and_b = [copy.deepcopy(model_layer.weight.data), copy.deepcopy(model_layer.bias.data)]
    
    # conv2d --> bit conv2d
    bitconv = pqmodel.BitConv2d(in_feature=model_layer.weight.data.size(1), 
                                out_feature=model_layer.weight.data.size(0),
                                kernel_size=model_layer.weight.data.size(2),
                                stride=1,
                                padding=1,
                                bias=True, 
                                bits_num=num_bits, 
                                n_scale=n_scale)
    
    bitconv.load_pretrain(w_pretrain=model_layer.weight.data, b_pretrain=model_layer.bias.data)    

    x = F.relu(bitconv(x))
    
    # only weights
    # model_layer.bias.data = fit_biases(model_layer.bias.data, s1, s2, mode=0)
    # x_only_weights = F.relu(model_layer(x)) # conv # use quantized weight to do convolution
    
    scale_factors_record = record_maxvalue(scale_factors_record, this_layer, x.max())
    
    # x, shift = bit_shift(x, keep_bits_num=num_bits) # bit-shift
    # shift = 0
    x = ScalebyM(x, M, shift=0) # mulipy M # Scaling

    return x, scale_factors_record

def fit_first_layer(x, num_bits=4, min_val=None, max_val=None):
    
    if not min_val and not max_val: 
        min_val, max_val = x.min(), x.max()
    qmin = 0.
    qmax = 2.**num_bits - 1.

    scale, zero_point = calcScaleZeroPoint(min_val, max_val, num_bits)
    
    q_x = zero_point + x / scale
    q_x.clamp_(qmin, qmax).round_()
    return q_x


def bit_conv_forward(x, model, num_bits, n_scale, val_record, scale_factors_record):

    """
      x : input datas.
      model : deep learning model.
      num_bits : quantized.
      val_record : dictionary to save min and max value.
    """
    
    weights_record = [] # to save float data (recover after forward)
    biases_record = []
    
    #     x = fit_first_layer(x, num_bits, stats['inputs']['ema_min'], stats['inputs']['ema_max'])


    #     """ conv1 """
    #     weights_record.append(model.conv1.weight.data)
    #     x = bit_conv_layer(x, 
    #                         model_layer=model.conv1,
    #                         stats=stats, num_bits=num_bits,
    #                         this_layer="conv1", prvs_layer="inputs", 
    #                         val_record=val_record)



    x = FakeQuantOp.apply(x, num_bits, stats['inputs']['ema_min'], stats['inputs']['ema_max'])
    """ conv1 """
    weights_record.append(model.conv1.weight.data)
    biases_record.append(model.conv1.bias.data)
    model.conv1.weight.data = FakeQuantOp_w.apply(model.conv1.weight.data, num_bits)
    x = F.relu(model.conv1(x))
    x = fit_first_layer(x, num_bits, stats['conv1']['ema_min'], stats['conv1']['ema_max'])
    
    
    """ conv2 """
    weights_record.append(model.conv2.weight.data)
    biases_record.append(model.conv2.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv2,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv2", prvs_layer="conv1", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    """ conv3 """
    weights_record.append(model.conv3.weight.data)
    biases_record.append(model.conv3.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv3,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv3", prvs_layer="conv2", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    """ conv4 """
    weights_record.append(model.conv4.weight.data)
    biases_record.append(model.conv4.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv4,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv4", prvs_layer="conv3", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    x = F.max_pool2d(x, 2, 2)
    
    """ conv5 """
    weights_record.append(model.conv5.weight.data)
    biases_record.append(model.conv5.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv5,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv5", prvs_layer="conv4", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    """ conv6 """
    weights_record.append(model.conv6.weight.data)
    biases_record.append(model.conv6.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv6,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv6", prvs_layer="conv5", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    """ conv7 """
    weights_record.append(model.conv7.weight.data)
    biases_record.append(model.conv7.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv7,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv7", prvs_layer="conv6", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    """ conv8 """
    weights_record.append(model.conv8.weight.data)
    biases_record.append(model.conv8.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv8,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv8", prvs_layer="conv7", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    x = F.max_pool2d(x, 2, 2)
    
    """ conv9 """
    weights_record.append(model.conv9.weight.data)
    biases_record.append(model.conv9.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv9,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv9", prvs_layer="conv8", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    """ conv10 """
    weights_record.append(model.conv10.weight.data)
    biases_record.append(model.conv10.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv10,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv10", prvs_layer="conv9", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    """ conv11 """
    weights_record.append(model.conv11.weight.data)
    biases_record.append(model.conv11.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv11,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv11", prvs_layer="conv10", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    """ conv12 """
    weights_record.append(model.conv12.weight.data)
    biases_record.append(model.conv12.bias.data)
    x, scale_factors_record = bit_conv_layer(x, 
                                        model_layer=model.conv12,
                                        stats=stats, num_bits=num_bits, n_scale=n_scale,
                                        this_layer="conv12", prvs_layer="conv11", 
                                        val_record=val_record,
                                        scale_factors_record=scale_factors_record)
    
    s = get_scale(stats["conv12"]['ema_min'], stats["conv12"]['ema_max'], num_bits)
    x = x*s
    x = F.max_pool2d(x, 2, 2)
    weights_record.append(model.classifier.weight.data)
    biases_record.append(model.classifier.bias.data)
    model.classifier.weight.data = FakeQuantOp_w.apply(model.classifier.weight.data, num_bits)
    x = model.classifier(x)
    x = x.view(-1, model.classes_num)
    
    return x, weights_record, biases_record, scale_factors_record
