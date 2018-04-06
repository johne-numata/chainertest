require 'pycall/import'
include PyCall::Import
pyimport 'pandas'
require 'csv'

PyCall.sys.path.append('C:\Users\D6113110\Desktop\chainer\LSTM実験')
pyimport 'lstm_class_1', as: :lstm

lstm.train()

