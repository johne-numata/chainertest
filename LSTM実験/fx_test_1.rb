# encoding: shift_jis
require 'csv'

csv_data =  CSV.read('GBPJPY.csv', headers:true)
train_master_data = []
TRAIN_SIZE = 30

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

p train_data[0]