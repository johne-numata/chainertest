require 'pycall/import'
include PyCall::Import
pyfrom 'chainer.datasets', import: :TupleDataset
require 'numpy'
np = Numpy
require 'csv'
require 'numo/narray'

PyCall.sys.path.append('C:\Users\D6113110\Desktop\chainer\LSTM実験')
pyimport 'lstm_class_2', as: :lstm

MINI_BATCH_SIZE = 100
LENGTH_OF_SEQUENCE = 100


#データファイルの読み込み
#csv_data =  CSV.read('GBPJPY.csv', headers:true)
#train_master_data = []

p table = CSV.table('GBPJPY.csv')

a = Numo::DFloat.new(table)

=begin
for i in 0...csv_data.size - 10
	# 以降10データ以内の最大小読み取り
	max = csv_data[i...i+10].map{|a| a[2]}.max.to_f / csv_data[i][4].to_f
	min = csv_data[i...i+10].map{|a| a[3]}.min.to_f / csv_data[i][4].to_f
	train_master_data << [csv_data[i][4].to_f, [max, min]]
end

train_data = []
for i in 0...train_master_data.size - TRAIN_SIZE
	train_data << train_master_data[i...i+TRAIN_SIZE]
end

# データの変換、numpy,tuple化
sequence = np.array(sequences, astype=np.float32)
t = np.array(t, astype=np.float32)
data = TupleDataset.(sequences, t)

lstm.train(data[0...(length/2).to_i], data[(length/2).to_i..-1], 100)
=end
