# Quantized model on chip.


### Train
float_train.py :  

	1. 訓練 float model
	2. 將 float model 中的 conv2d 和 bn 合併
	3. 訓練quantized model並存出 weights 和 bias 以及每一層的最大值最小值(scale factor)
   
### Inference
quan_forward.py  

	測試bit_shift效果



postquan_forward.py

	bit分開進行convolution然後scale之後再加總

