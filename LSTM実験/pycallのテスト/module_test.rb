require 'pycall/import'
include PyCall::Import
pyimport 'pandas'

#PyCall.sys.path.append(__dir__) ←__dir__の解釈がうまくいかない様子
PyCall.sys.path.append('C:\Users\D6113110\Desktop\chainer\LSTM実験\pycallのテスト')

pyimport 'module_test', as: :mt

mt.p_list(7)
