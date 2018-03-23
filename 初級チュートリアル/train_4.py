from chainer.datasets import cifar
from chainer import iterators
import chainer
import chainer.links as L
import chainer.functions as F

class MyNet(chainer.Chain):
	def __init__(self, n_out):
		super(MyNet, self).__init__()
		with self.init_scope():
			self.conv1 = L.Convolution2D(None, 32, 3, 3, 1)
			self.conv2 = L.Convolution2D(32, 64, 3, 3, 1)
			self.conv3 = L.Convolution2D(64, 128, 3, 3, 1)
			self.fc4 = L.Linear(None, 1000)
			self.fc5 = L.Linear(1000, n_out)
	
	def __call__(self, x):
		h = F.relu(self.conv1(x))
		h = F.relu(self.conv2(h))
		h = F.relu(self.conv3(h))
		h = F.relu(self.fc4(h))
		return self.fc5(h)

class ConvBlock(chainer.Chain):
	def __init__(self, n_ch, pool_drop = False):
		w = chainer.initializers.HeNormal()
		super(ConvBlock, self).__init__()
		with self.init_scope():
			self.conv = L.Convolution2D(None, n_ch, 3, 1, 1, nobias= True, initialW= w)
			self.bn = L.BatchNormalization(n_ch)
		self.pool_drop = pool_drop

	def __call__(self, x):
		h = F.relu(self.bn(self.conv(x)))
		if self.pool_drop:
			h = F.max_pooling_2d(h, 2, 2)
			h = F.dropout(h, ratio= 0.25)
		return h

class LinearBlock(chainer.Chain):
	def __init__(self, drop= False):
		w = chainer.initializers.HeNormal()
		super(LinearBlock, self).__init__()
		with self.init_scope():
			self.fc = L.Linear(None, 1024, initialW = w)
		self.drop = drop

	def __call__(self, x):
		h = F.relu(self.fc(x))
		if self.drop:
			h = F.dropout(h)
		return h

class DeepCNN(chainer.ChainList):
	def __init__(self, n_output):
		super(DeepCNN, self).__init__(
			ConvBlock(64),
			ConvBlock(64, True),
			ConvBlock(128),
			ConvBlock(128, True),
			ConvBlock(256),
			ConvBlock(256),
			ConvBlock(256),
			ConvBlock(256, True),
			LinearBlock(),
			LinearBlock(),
			L.Linear(None, n_output)
		)

	def __call__(self, x):
		for f in self:
			x = f(x)
		return x

from chainer import training
from chainer import optimizers
import matplotlib.pyplot as plt
import matplotlib
from chainer.training import extensions

def train(network_object, batchsize=128, max_epoch=20, train_dataset=None,
			test_dataset=None, postfix='', base_lr=0.01, lr_decay=None):

	# Dataset
	if train_dataset is None and test_dataset is None:
		train, test = cifar.get_cifar10()
	else:
		train, test = train_dataset, test_dataset

	# Iterator
	batchsize = 128
	train_iter = iterators.SerialIterator(train, batchsize)
	test_iter = iterators.SerialIterator(test, batchsize, repeat = False, shuffle = False)

	# Model
	net = L.Classifier(network_object)

	# Optimizer
	optimizer = optimizers.MomentumSGD(lr = base_lr)
	optimizer.setup(net)
	optimizer.add_hook(chainer.optimizer.WeightDecay(0.0005))

	# Updater
	updater = training.StandardUpdater(train_iter, optimizer)

	# Triner
	trainer = training.Trainer(updater, (max_epoch, 'epoch'), 
				out='{}_cifer10_{}result'.format(network_object.__class__.__name__, postfix))

	# Trainer extension
	trainer.extend(extensions.LogReport())
	trainer.extend(extensions.observe_lr())
	trainer.extend(extensions.Evaluator(test_iter, net), name= 'val')
	trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/accuracy',
			'val/main/loss', 'val/main/accuracy', 'elapsed_time', 'lr']))
	trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'],
			 x_key = 'epoch', file_name= 'loss.png'))
	trainer.extend(extensions.PlotReport(['main/accuracy', 'val/main/accuracy'],
			 x_key = 'epoch', file_name= 'accuracy.png'))
	if lr_decay is not None:
		trainer.extend(extensions.ExponentialShift('lr', 0.1), trigger= lr_decay)

	trainer.run()
	del trainer

	return net

#net = train(MyNet(10))
model = train(DeepCNN(10), max_epoch= 100, base_lr= 0.1, lr_decay= (30, 'epoch'))

cls_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
				'dog', 'frog', 'horse', 'ship', 'truck']

def predict(net, image_id):
	_, test = cifar.get_cifar10()
	x, t = test[image_id]
	y = net.predictor(x[None, ...]).data.argmax(axis= 1)[0]
	print('predicted_label:', cls_names[y])
	print('answer:', cls_names[t])

	plt.imshow(x.transpose(1, 2, 0))
	plt.show()

for i in range(10, 15):
	predict(net, i)
	
