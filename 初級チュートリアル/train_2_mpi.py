import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions
from chainer.datasets import mnist
from chainer import iterators, training, optimizers
import chainermn
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
net = L.Classifier(net)

comm = chainermn.create_communicator('naive')
if comm.rank == 0:
	train, test = mnist.get_mnist(withlabel = True, ndim = 1)
else:
	train, test = None, None

batchsize = 64
train = chainermn.scatter_dataset(train, comm, shuffle=False)
test = chainermn.scatter_dataset(test, comm, shuffle=False)
train_iter = iterators.SerialIterator(train, batchsize)
test_iter = iterators.SerialIterator(test, batchsize, repeat = False, shuffle = False)

optimizer = chainermn.create_multi_node_optimizer(optimizers.SGD(lr=0.01), comm)
optimizer.setup(net)

updater = training.StandardUpdater(train_iter, optimizer)

max_epoch = 20

trainer = training.Trainer(updater, (max_epoch, 'epoch'), out='mnist_result')

if comm.rank == 0:
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.Evaluator(test_iter, net), name='val')
	trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy', 
				'val/main/loss', 'val/main/accuracy', 'elapsed_time']))

trainer.run()
