# http://seiya-kumada.blogspot.jp/2016/07/lstm-chainer.html

import chainer
import chainer.links as L
import chainer.functions as F 
from chainer import Variable, iterators, report
from chainer.datasets import TupleDataset
import numpy as np
import math
import random
from chainer import training
from chainer.training import extensions

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
		h = Variable(np.asarray([_h[-1].data for _h in h], dtype=np.float32))
		h = F.relu(self.l2(h))
		y = self.l3(h)
		self.loss = F.mean_squared_error(y, t)
		report({'loss': self.loss}, self)
		return self.loss


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
            sequences.append(np.asarray(all_data[i:i+length_of_sequence], dtype=np.float32))
            t.append(np.asarray(all_data[i+length_of_sequence], dtype=np.float32))
        return TupleDataset(sequences, t)


# Updater改変
class MyUpdater(training.StandardUpdater):

    def __init__(self, train_iter, optimizer):
        super(MyUpdater, self).__init__(train_iter, optimizer)

    def update_core(self):
        batch = self.get_iterator('main').next()
        in_arrays = self.converter(batch, self.device)
        optimizer = self.get_optimizer('main')
        a = [Variable(x[:, np.newaxis].astype(np.float32)) for x in in_arrays[0]]
        b = Variable(in_arrays[1][:, np.newaxis])

        loss = optimizer.target(a, b)
        optimizer.target.cleargrads()
        loss.backward()
        optimizer.update()

#        import pdb; pdb.set_trace()
#        loss_func = optimizer.target()
#        optimizer.update(optimizer.target(a, b))


from chainer import optimizers, training
from chainer.training import extensions
from chainer import reporter as reporter_module
import time
import sys

IN_UNITS = 1
HIDDEN_UNITS = 5
OUT_UNITS = 1
TRAINING_EPOCHS = 1000
DISPLAY_EPOCH = 10
MINI_BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 100
STEPS_PER_CYCLE = 50
NUMBER_OF_CYCLES = 100

if __name__ == "__main__":

    # make training data
    data_maker = DataMaker(steps_per_cycle=STEPS_PER_CYCLE, number_of_cycles=NUMBER_OF_CYCLES)
    train_data = data_maker.make(LENGTH_OF_SEQUENCE)
	# Iterator
    batchsize = 100
    train_iter = iterators.SerialIterator(train_data, batchsize)
#    import pdb; pdb.set_trace()

    # setup model
    model = LSTM(IN_UNITS, HIDDEN_UNITS, OUT_UNITS)
 
    # setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
 
    start = time.time()
    
    updater = MyUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.observe_lr())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'elapsed_time', 'lr']))
    trainer.run()

    end = time.time()
 
    print("{}[sec]".format(end - start))

