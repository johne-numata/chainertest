import numpy as np
import chainer
from chainer import Function, gradient_check, report, training, utils, Variaable
from chainer import datasets, iterators, optimizers, selializers
from chainer import Link, Chain, ChinList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

Layer = 1

class MyLSTM(Chain):
	def __init__(self, n_in, n_mid, n_out):
		super(MyLSTM, self).__init__()
		with self.init_scope():
			self.l1 = L.NStepLSTM(self, Layer, n_in, n_mid, dropout=0.5)
			self.l2 = L.Liner(None, 20)
			self.l3 = L.Liner(None, n_out)

	def __call__(self, x, t):
		self.l1.reset_state()
		_, _, y = self.l1(x)
		y = self.l2(y)
		y = self.l3(y)
		return F.mean_squared_error(y, t)

model = MyLSTM(1, 1)
optimizer = optimizers.Adam()
optimizer.setup(model)






