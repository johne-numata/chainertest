from chainer.datasets import mnist
train, test = mnist.get_mnist(withlabel = True, ndim = 1)

from chainer import iterators
batchsize = 128
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, repeat = False, shuffle = False)

import chainer
import chainer.links as L
import chainer.functions as F

import random
import numpy as np
random.seed(0)
np.random.seed(0)

class MLP(chainer.Chain):
	def __init__(self, n_mid_units=100, n_out=10):
		super(MLP, self).__init__()
		with self.init_scope():
			self.l1 = L.Linear(None, n_mid_units)
			self.l2 = L.Linear(n_mid_units, n_mid_units)
			self.l3 = L.Linear(n_mid_units, n_out)
	
	def __call__(self, x):
		h1 = F.relu(self.l1(x))
		h2 = F.relu(self.l2(h1))
		return self.l3(h2)

net = MLP()

from chainer import training
net = L.Classifier(net)

from chainer import optimizers
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(net)

updater = training.StandardUpdater(train_iter, optimizer)

max_epoch = 20

trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

import matplotlib.pyplot as plt
import matplotlib
from chainer.training import extensions
trainer.extend(extensions.LogReport())
#trainer.extend(extensions.snapshot(filename='snapshot_epoch-{.updater.epoch}'))
trainer.extend(extensions.Evaluator(test_iter, net), name='val')
trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 
				'val/main/loss', 'val/main/accuracy', 'elapsed_time']))
#trainer.extend(extensions.PlotReport(['l1/W/data/std'], x_key='epoch', file_name='loss.png'))
#trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key='epoch', file_name='std.png'))
#trainer.extend(extensions.dump_graph('main/loss'))

trainer.run()
