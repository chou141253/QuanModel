import torch
import torch.nn as nn
import math


class BitConv2d(nn.Module):

    """
    this layer only call under inferencing.
    """

    def __init__(self, 
                 in_feature, 
                 out_feature,
                 kernel_size,
                 stride,
                 padding,
                 bias, 
                 bits_num, 
                 n_scale):
        """
        in_features: input channel number
        out_features: output channel number
        kernel_size: can be int or tuple
        stride: step of convolution
        padding: padding value
        bits_num: the bits number of input and weight.
        n_scale: int k. discord precision by 1/k.
        """

        self.bits_num = bits_num # loop
        self.n_scale = n_scale # scale factor. experiance key.
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.buffer = torch.zeros([bits_num], dtype=torch.float32)

        self.conv = Conv2d(in_features, out_features, kernel_size, stride, padding, bias=False)
        self.bias = bias

    def load_pretrain(self, w_pretrain, b_pretrain=None):
        self.conv.weight.data = w_pretrain
        if self.bias is not None:
            self.bias = b_pretrain

    def separate_to_bits(self, x, N, is_after=True, log=False):
        """
        see print_binary.py to find detial.
        is_after: 最後在乘以 2^(bit_num)嗎? (bits shift).
        """

        bit_x = torch.zeros([self.bits_num, x.size(0), x.size(1), x.size(2), x.size(3)],dtype=torch.float32)

        if log: 
            print("start separating..")
        down_factor = 2**self.bits_num
        for i in range(self.bits_num):
            if log:
                print("    building {}(int) {}(binary)"\
                    .format(down_factor, bin(down_factor).split('b')[-1].zfill(self.bits_num)))
            
            if is_after:
                tmp_bit = self.quantized((x/down_sample == 1).type(torch.int32)/N)*(2**(down_factor))
            else:
                tmp_bit = self.quantized((x/down_sample == 1).type(torch.int32)*(2**(down_factor))/N)

            bit_x[self.bits_num-1-i] = tmp_bit
            x = x - tmp_bit*down_factor
            down_factor = 2**(self.bits_num-1-i)

        if log:
            print("separate done.")

        return bit_x

    def quantized(self, f_x):
        """
        different between quan-model and postquan-model,
        because this only call when inference.
        """
        int_x = f_x.type(torch.int32)
        int_x = int_x.type(torch.float32)
        return int_x

    # convolutional
    def forward(self, x):

        batch_size = x.size(0)

        x = self.separate_to_bits(x, N=self.n_scale, is_after=True, log=False)
        # out size batch out_features
        postquan_x = torch.zeros([batch_size, self.out_features, x.size(2), x.size(3)], dtype=torch.float32)
        #conv
        for bit_id in range(self.bits_num):
            postquan_x += self.conv(x[bit_id])

        postquan_x += self.bias

        return postquan_x

