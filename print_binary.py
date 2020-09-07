import torch
import json

for i in range(33):
	read_b = (bin(i).split('b')[-1]).zfill(6)
	print("{}  ---->  {}".format(str(i).zfill(3), read_b))

	bits = ""
	value = i
	# step1.
	best_fit = int(value/32)
	bits += str(best_fit)
	value -= 32*best_fit
	# step2.
	best_fit = int(value/16)
	bits += str(best_fit)
	value -= 16*best_fit
	# step3.
	best_fit = int(value/8)
	bits += str(best_fit)
	value -= 8*best_fit
	# step4.
	best_fit = int(value/4)
	bits += str(best_fit)
	value -= 4*best_fit
	# step5.
	best_fit = int(value/2)
	bits += str(best_fit)
	value -= 2*best_fit
	# step6.
	best_fit = int(value/1)
	bits += str(best_fit)
	value -= 1*best_fit

	print("           ", bits)
