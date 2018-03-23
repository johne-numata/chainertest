from __future__ import print_function
import sys
import numpy as np
from model import Model
from data import NData

def main(f_model, f_on, f_off):
	data = NData.load_data(f_on, f_off)
	n = len(data)
	units = len(data[0][0])
	model = Model(units)
	model.load(f_model)
	s = 0
	f = open('failed.txt', 'w')
	for i in range(n):
		x = np.array([data[i][0]], dtype=np.float32)
		y = model.predictor(x).data[0]
		if np.argmax(y) == data[i][1]:
			s = s + 1
		else:
			f.write(str(data[i][0]) + '\n')
	c = float(s) / float(n)
	print('Success/Total = %d/%d' % (s, n))
	print('Ratio = %f' % c)

if __name__ == "__main__":
	main("test.model", "on_test.txt", "off_test.txt")
