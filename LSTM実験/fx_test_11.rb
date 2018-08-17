require 'pycall/import'
include PyCall::Import
pyfrom 'chainer.datasets', import: :TupleDataset
require 'numpy'
np = Numpy
require 'csv'
require 'numo/narray'

#PyCall.sys.path.append(__dir__)
PyCall.sys.path.append('C:\Users\D6113110\Desktop\chainer\LSTM実験')
pyimport 'lstm_class_2', as: :lstm

BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 250
LENGTH_OF_VERIFI = 10
EPOCHS = 500
EMA_1 = 25		# 5,10,21,25
MACD_S =		# 12,26,9
MACD_L =
STCAS = 

#EMA計算
def ema(data)
	d = data[true, 3]
	ema = Numo::DFloat.zeros(data.shape[0])
	ema[EMA_1 - 1] = (d[0...EMA_1].sum + d[EMA_1 - 1]) / (EMA_1 + 1)
	for i in EMA_1...data.shape[0]
		ema[i] = d[i] * 2.0 / (EMA_1 + 1) + ema[i - 1] * (1.0 - 2.0 / (EMA_1 + 1))
	end
	return data.concatenate(ema.expand_dims(1), axis:1)
end

#データファイルの読み込み
csv_data =  CSV.read('GBPJPY.csv', headers:true, converters: :numeric)
#csv_data =  CSV.read('bardata.csv', headers:true, converters: :numeric)
#data = csv_data.by_col[1..-1].transpose
data = Numo::NArray[*csv_data.by_col[1..-1]]
data = ema(data)[EMA_1..-1, true]

sequences = []
t = []
for i in 0...data.shape[0] - LENGTH_OF_SEQUENCE - LENGTH_OF_VERIFI
	sequences << (data[i...i + LENGTH_OF_SEQUENCE, true] / data[i + LENGTH_OF_SEQUENCE - 1, 3] - 1.0).to_a
	 # 以降10データ内の 上げor下げ幅/下げor上げ幅　上げが大きい場合が+の値
	max = ([data[i + LENGTH_OF_SEQUENCE..i + LENGTH_OF_SEQUENCE + LENGTH_OF_VERIFI, 1].max, data[i + LENGTH_OF_SEQUENCE - 1, 3]].max \
			- data[i + LENGTH_OF_SEQUENCE - 1, 3]) / data[i + LENGTH_OF_SEQUENCE - 1, 3]
	min = ([data[i + LENGTH_OF_SEQUENCE..i + LENGTH_OF_SEQUENCE + LENGTH_OF_VERIFI, 2].min, data[i + LENGTH_OF_SEQUENCE - 1, 3]].min \
			- data[i + LENGTH_OF_SEQUENCE - 1, 3]) / data[i + LENGTH_OF_SEQUENCE - 1, 3]
    if max.abs > min.abs
#		v = max
    	min == 0 ? v = max : v = (max/min).abs * max
    	v = 2 if v > 2
    else
#		v = min
    	max == 0 ? v = min : v = (min/max).abs * min
    	v = -2 if v < -2
    end
    t << v
end

#p sequences[-2..-1]
#p t[-2..-1]
#p sequences.size
#p t.size

=begin
# データの変換、numpy,tuple化
sequence = np.array(sequences, astype=np.float32)
t = np.array(t, astype=np.float32)
data = TupleDataset.(sequences, t)
# 解析実行
#lstm.train(data[PyCall::Slice.(0,-1,2)], data[PyCall::Slice.(1,-1,2)], 128, 1, 4, 20, 1)
#lstm.train(data[0...(sequences.size/2).to_i], data[(sequences.size/2).to_i..-1], 128, 1, 1, 20, 1)
=end
#=begin
# データの変換、numpy,tuple化
train_s = []; train_t = []; test_s = []; test_t = []
for i in 0...data.shape[0] - LENGTH_OF_SEQUENCE - LENGTH_OF_VERIFI
	if i % 5 == 0
 		test_s << sequences[i]
 		test_t << t[i]
 	else
		train_s << sequences[i]
		train_t << t[i]
 	end
end
train_data = TupleDataset.(np.array(train_s, astype=np.float32), np.array(train_t, astype=np.float32))
test_data = TupleDataset.(np.array(test_s, astype=np.float32), np.array(test_t, astype=np.float32))
# 解析実行
lstm.train(train_data, test_data, BATCH_SIZE, EPOCHS, 1, 5, 20, 1)
#=end
