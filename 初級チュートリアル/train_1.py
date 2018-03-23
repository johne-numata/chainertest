from chainer.datasets import mnist
train, test = mnist.get_mnist(withlabel = True, ndim = 1)

import matplotlib.pyplot as plt
x, t = train[0]
plt.imshow(x.reshape(28, 28), cmap = 'gray')
plt.axis('off')
plt.show()
print('label:', t)

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

from chainer import optimizers
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(net)

from chainer.dataset import concat_examples

max_epoch = 10

while train_iter.epoch < max_epoch:
	train_batch = train_iter.next()
	x, t = concat_examples(train_batch) 
	y = net(x)
	loss = F.softmax_cross_entropy(y, t)
	net.cleargrads()
	loss.backward()
	optimizer.update()
	
	if train_iter.is_new_epoch:
		print('epoch:{:02d} train_loss:{:.04f} '.format(
			train_iter.epoch, float(loss.data)), end='')
		test_losses = []
		test_accuracies = []
		while True:
			test_batch = test_iter.next()
			x_test, t_test = concat_examples(test_batch)
			y_test = net(x_test)
			loss_test = F.softmax_cross_entropy(y_test, t_test)
			test_losses.append(loss_test.data)
			accuracy = F.accuracy(y_test, t_test)
			test_accuracies.append(accuracy.data)
			
			if test_iter.is_new_epoch:
				test_iter.epoch = 0
				test_iter.current_position = 0
				test_iter.is_new_epoch = False
				test_iter._pushed_position = None
				break
		print('val_loss:{:.04f} val_accuracy:{:.04f}'.format(
			np.mean(test_losses), np.mean(test_accuracies)))

from chainer import serializers
serializers.save_npz('my_mnist.model', net)
