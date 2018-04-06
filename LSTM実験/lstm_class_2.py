import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, report, training, utils
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions
import numpy as np


# モデル定義
class LSTM(chainer.Chain):

	def __init__(self, in_units=1, hidden_units=2, out_units=1):
		super(LSTM, self).__init__(
			l1=L.NStepLSTM(1, in_units, hidden_units, 0.3),
			l2=L.Linear(hidden_units, hidden_units),
			l3=L.Linear(hidden_units, out_units),
		)

	def __call__(self, x, t):
#		import pdb; pdb.set_trace()
		_, _, h = self.l1(None, None, x)
		h = F.stack([F.get_item(_h, -1) for _h in h])
		h = F.relu(self.l2(h))
		y = self.l3(h)
		self.loss = F.mean_squared_error(y, t)
		report({'loss': self.loss}, self)
		return self.loss


# converter改変
def MyConverter(batch, device=None):
    x = [Variable(np.asarray(data[0]).astype(np.float32)[:, np.newaxis]) for data in batch]
    t = Variable(np.asarray([data[1] for data in batch]).astype(np.float32)[:, np.newaxis])
    return x, t


# Train実行
def train(x_data, t_data, batchsize=128, layer=1, in_units=1, hidden_units=5, out_units=1):

	# Iterator
    batchsize = batchsize
    train_iter = iterators.SerialIterator(x_data, batchsize)
    test_iter = iterators.SerialIterator(t_data, batchsize, repeat = False, shuffle = False)

    # setup model
    model = LSTM(in_units, hidden_units, out_units)

    # setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    updater = training.StandardUpdater(train_iter, optimizer, MyConverter)
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.Evaluator(test_iter, model, MyConverter), name= 'val')
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'val/main/loss', 'elapsed_time', 'lr']))
    trainer.extend(extensions.PlotReport(['main/loss', 'val/main/loss'], x_key = 'epoch', file_name= 'loss.png'))
#    trainer.extend(extensions.ProgressBar())

    trainer.run()

