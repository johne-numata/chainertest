import chainer
import numpy as np

class NData:
	@staticmethod
	def load_data(f_on, f_off):
		data = []
		for line in open(f_on, 'r'):
			s = line.split(",")
			n = [float(t) for t in s]
			data.append([n, 1])
		for line in open(f_off, "r"):
			s = line.split(",")
			n = [float(t) for t in s]
			data.append([n, 0])
		return data
	
	@staticmethod
	def make_dataset(data):
		n = len(data)
		xn = len(data[0][0])
		x = np.empty((n, xn), dtype=np.float32)
		y = np.empty(n, dtype=np.int32)
		for i in range(n):
			x[i] = np.asarray(data[i][0])
			y[i] = data[i][1]
		return chainer.datasets.TupleDataset(x, y)
