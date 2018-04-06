import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable, report, training, utils
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.training import extensions
import numpy as np
import math
import random
from chainer.datasets import TupleDataset

random.seed(0)
np.random.seed(0)


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
#def train(x_data, t_data, btchsize=128, layer=1, in_units=1, hidden_units=5, out_unit=1):
def train():
    # make training data
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)
    data = data_maker.make(LENGTH_OF_SEQUENCE)
    harf = len(data) // 2
    train_data = data[:harf]
    test_data = data[harf:]
	# Iterator
    batchsize = 100
    train_iter = iterators.SerialIterator(train_data, batchsize)
    test_iter = iterators.SerialIterator(test_data, batchsize, repeat = False, shuffle = False)

    # setup model
    model = LSTM(IN_UNITS, HIDDEN_UNITS, OUT_UNITS)

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


# データ作成クラス
class DataMaker(object):

    def __init__(self, steps_per_cycle, number_of_cycles):
        self.steps_per_cycle = steps_per_cycle
        self.number_of_cycles = number_of_cycles

    def make(self, length_of_sequence):
        all_data = np.array([math.sin(i * 2 * math.pi/self.steps_per_cycle) for i in range(self.steps_per_cycle)] * self.number_of_cycles)

        sequences = []
        t = []
        for i in range(len(all_data) - length_of_sequence):
            sequences.append(all_data[i:i+length_of_sequence])
            t.append(all_data[i+length_of_sequence])
        return TupleDataset(sequences, t)

IN_UNITS = 1
HIDDEN_UNITS = 5
OUT_UNITS = 1
TRAINING_EPOCHS = 1000
DISPLAY_EPOCH = 10
MINI_BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 100
STEPS_PER_CYCLE = 50
NUMBER_OF_CYCLES = 100

