# chainerと必要なパッケージをインポート
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable, datasets, optimizers
from chainer import report, training
from chainer.training import extensions
import chainer.cuda
import numpy as np
from chainer.dataset import concat_examples

# フィボナッチ数列を割る値
VALUE = 5

# 時系列データの全長
TOTAL_SIZE = 2000

# フィボナッチ数列から周期関数を作る
class DatasetMaker(object):
	@staticmethod
	def make(total_size, value):
		return (DatasetMaker.fibonacci(total_size) % value).astype(np.float32)

	# 全データを入力時のシーケンスに分割する
	@staticmethod
	def make_sequences(data, seq_size):
		data_size = len(data)
		row = data_size - seq_size
		seq = np.ndarray((row, seq_size)).astype(np.float32)
		for i in range(row):
			seq[i, :] = data[i: i + seq_size]
		return seq

	@staticmethod
	def fibonacci(size):
		values = [1, 1]
		for _ in range(size - len(values)):
			values.append(values[-1] + values[-2])
		return np.array(values)


# データの作成
dataset = DatasetMaker.make(TOTAL_SIZE, VALUE)




#訓練データと教師データへの分割
 
x,t = [],[]
 
N = len(dataset)
M = 25
for n in range(M,N):
  _x = dataset[n-M:n]
  _t = dataset[n]
  x.append(_x)
  t.append(_t)
 
x = Variable(np.array(x, dtype = np.float32))
t = Variable(np.array(t, dtype = np.float32).reshape(len(t),1))
 
# 訓練：60%, 検証：40%で分割する
n_train = int(len(x) * 0.6)
dataset = list(zip(x, t))
train, test = chainer.datasets.split_dataset(dataset, n_train)





# ニューラルネットワークモデルを作成
class RNN(Chain):
	def __init__(self, n_input, n_units, n_output):
		super().__init__()
		with self.init_scope():
			self.l1 = L.NStepLSTM(n_layers=1, in_size=n_input, out_size=n_units, dropout=0.3)
			self.l2 = L.Linear(None, n_output)

	def __call__(self, x, t):
		_, _, h = self.l1(None, None, x)
		h = Variable(np.asarray([_h[-1].data for _h in h], dtype=np.float32))
		y = self.l2(h)
		loss = F.mean_squared_error(y, t)
		report({'loss':loss},self)
		return loss


# 乱数のシードを固定 (再現性の確保)
np.random.seed(1)

# モデルの宣言
model = RNN(1, 30, 1)

# Optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)
 
# Iterator
batchsize = 50
train_iter = chainer.iterators.SerialIterator(train, batchsize)
test_iter = chainer.iterators.SerialIterator(test, batchsize, repeat=False, shuffle=False)
 
# Updater &lt;- LSTM用にカスタマイズ
#updater = training.StandardUpdater(train_iter, optimizer)

# Trainerとそのextensions
epoch = 3000
while train_iter.epoch < epoch:
	train_batch = train_iter.next()
	import pdb; pdb.set_trace()
	x, t = concat_examples(train_batch)
	train = model(x, t)
	model,cleargrads()
	loss.backwrd()
	optimizer.update()

#trainer = training.Trainer(updater, (epoch, 'epoch'), out='result')
 
# 評価データで評価
#trainer.extend(extensions.Evaluator(test_iter, model,device = -1))
 
# 学習結果の途中を表示する
#trainer.extend(extensions.LogReport(trigger=(1, 'epoch')))
 
# １エポックごとに、trainデータに対するlossと、testデータに対するlossを出力させる
#trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss', 'elapsed_time']), trigger=(1, 'epoch'))


#trainer.run()

