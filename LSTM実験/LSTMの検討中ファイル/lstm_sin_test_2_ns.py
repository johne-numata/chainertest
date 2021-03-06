﻿# http://seiya-kumada.blogspot.jp/2016/07/lstm-chainer.html

import chainer
import chainer.links as L
import chainer.functions as F 
from chainer import Variable

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
		return self.loss


import numpy as np
import math
import random
 
random.seed(0)
 
class DataMaker(object):
 
    def __init__(self, steps_per_cycle, number_of_cycles):
        self.steps_per_cycle = steps_per_cycle
        self.number_of_cycles = number_of_cycles
 
    def make(self):
        return np.array([math.sin(i * 2 * math.pi/self.steps_per_cycle) for i in range(self.steps_per_cycle)] * self.number_of_cycles)
 
    def make_mini_batch(self, data, mini_batch_size, length_of_sequence):
        sequences = []
        t = np.ndarray(mini_batch_size, dtype=np.float32)
#        import pdb; pdb.set_trace()
        for i in range(mini_batch_size):
            index = np.random.randint(0, len(data) - length_of_sequence)
            sequences.append( Variable(np.asarray(data[index:index+length_of_sequence], dtype=np.float32)[:, np.newaxis]))
            t[i] = data[index+length_of_sequence]
        return sequences, t

from chainer import optimizers
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
    train_data = data_maker.make()
 
    # setup model
    model = LSTM(IN_UNITS, HIDDEN_UNITS, OUT_UNITS)
 
    # setup optimizer
    optimizer = optimizers.Adam()
    optimizer.setup(model)
 
    start = time.time()
    cur_start = start
    for epoch in range(TRAINING_EPOCHS):
        sequences, t = data_maker.make_mini_batch(train_data, mini_batch_size=MINI_BATCH_SIZE, length_of_sequence=LENGTH_OF_SEQUENCE)
        model.cleargrads()
        loss = model(sequences, Variable(t[:, np.newaxis]))
#        import pdb; pdb.set_trace()
        loss.backward()
#        loss.unchain_backward()  # NStepLSTMでは不要
        optimizer.update()
 
        if epoch != 0 and epoch % DISPLAY_EPOCH == 0:
            cur_end = time.time()
            # display loss
            print(
                "[{j}]training loss:\t{i}\t{k}[sec/epoch]".format(
                    j=epoch, 
                    i=loss.data/(len(sequences[0].data) - 1), 
                    k=(cur_end - cur_start)/DISPLAY_EPOCH
                )
            )
            cur_start = time.time() 
            sys.stdout.flush()
 
    end = time.time()
 
    print("{}[sec]".format(end - start))

