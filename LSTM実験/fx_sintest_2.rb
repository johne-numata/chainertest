require 'pycall/import'
include PyCall::Import
pyfrom 'chainer.datasets', import: :TupleDataset
require 'numpy'
np = Numpy

PyCall.sys.path.append('C:\Users\D6113110\Desktop\chainer\LSTM実験')
pyimport 'lstm_class_2', as: :lstm

MINI_BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 100
STEPS_PER_CYCLE = 50
NUMBER_OF_CYCLES = 100

# データ作成
all_data = (0...STEPS_PER_CYCLE).to_a.collect{|i| Math.sin(i * 2 * Math::PI/STEPS_PER_CYCLE)} * NUMBER_OF_CYCLES
length = all_data.size
sequences = []
t = []
(length - LENGTH_OF_SEQUENCE).times{|i|
	sequences << all_data[i...i+LENGTH_OF_SEQUENCE]
	t << all_data[i+LENGTH_OF_SEQUENCE]
}

# データ加工(numpy,tuple化)
sequence = np.array(sequences, astype=np.float32)
t = np.array(t, astype=np.float32)
data = TupleDataset.(sequences, t)

lstm.train(data[0...(length/2).to_i], data[(length/2).to_i..-1], 100)
