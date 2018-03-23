import chainer
from chainer import optimizers
import chainer.links as L
import chainer.functions as F
import numpy as np
from chainer import serializers
import _pickle

# フィボナッチ数列を割る値
VALUE = 5

# 常に同じ計算をする
np.random.seed(0)

# 時系列データの全長
TOTAL_SIZE = 2000

# 訓練とテストの分割比
SPRIT_RATE = 0.9

# 入力時の時系列データ長
SEQUENCE_SIZE = 50

EPOCHS = 30
BATCH_SIZE = 100

# 入力層の次元
N_IN = 1

# 隠れ層の次元
N_HIDDEN = 200

# 出力層の次元
N_OUT = 1

# LSTM
class MyNet(chainer.Chain):
	def __init__(self, n_in=1, n_hidden=20, n_out=1, train=True):
		super(MyNet, self).__init__()
		with self.init_scope():
			self.l1 = L.LSTM(n_in, n_hidden, lateral_init=chainer.initializers.Normal(scale=0.01))
			self.l2 = L.Linear(n_hidden, n_out, initialW=chainer.initializers.Normal(scale=0.01))
			self.train = train

	def __call__(self, x):			
		with chainer.using_config('train', self.train):
			h = self.l1(x)
			y = self.l2(h)
		return y

	def reset_state(self):
		self.l1.reset_state()

# 損失値計算機
class LossCalculator(chainer.Chain):
	def __init__(self, model):
		super(LossCalculator, self).__init__()
		with self.init_scope():
			self.model = model

	def __call__(self, x, t):
		y = self.model(x)
		loss = F.mean_squared_error(y, t)
		return loss

# バッチ単位で一つのシーケンスを学習する	
def calculate_loss(model, seq):
	rows, cols = seq.shape
	assert cols - 1 == SEQUENCE_SIZE
	loss = 0
	# 1つのシーケンスを全て計算する。
	for i in range(cols - 1):
		# Batch単位で計算
		x = chainer.Variable(
			np.asarray([seq[j, i + 0] for j in range(rows)], dtype=np.float32)[:, np.newaxis]
		)
		t = chainer.Variable(
			np.asarray([seq[j, i + 1] for j in range(rows)], dtype=np.float32)[:, np.newaxis]
		)
		# 誤差を蓄積する
		loss += model(x, t)

	return loss

# モデルを更新する
def update_model(model, seq):
	loss = calculate_loss(model, seq)
	#誤差逆伝播
	loss_calculator.cleargrads()
	loss.backward()

	# バッチ単位で古い記憶を削除し、再計算コストを削減する。
	loss.unchain_backward()

	# バッチ単位で更新する
	optimizer.update()
	return loss

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

# テストデータに対する誤差を計算する
def evaluate(loss_calculator, seqs):
	batches = seqs.shape[0] // BATCH_SIZE
	clone = loss_calculator.copy()
	clone.train = False
	clone.model.reset_state()
	start = 0
	for i in range(batches):
		seq = seqs[start: start + BATCH_SIZE]
		start += BATCH_SIZE
		loss = calculate_loss(clone, seq)
	return loss

if __name__ == '__main__':
	# データの作成
	dataset = DatasetMaker.make(TOTAL_SIZE, VALUE)
	# 訓練データと検証データに分ける
	n_train = int(TOTAL_SIZE * SPRIT_RATE)
	n_val = TOTAL_SIZE - n_train
	train_dataset = dataset[: n_train].copy()
	val_dataset = dataset[n_train:].copy()
	# 長さSEUENCE_SIZE + 1の時系列データを始点を1つずつずらして作る
	# +1は教師データ
	train_seq = DatasetMaker.make_sequences(train_dataset, SEQUENCE_SIZE + 1)
	val_seq = DatasetMaker.make_sequences(val_dataset, SEQUENCE_SIZE + 1)
	
	#モデルの設定
	mynet = MyNet(N_IN, N_HIDDEN, N_OUT)
	loss_calculator = LossCalculator(mynet)
	
	# 最適化器の作成
	optimizer = optimizers.Adam()
	optimizer.setup(loss_calculator)
	
	# 訓練
	batches = train_seq.shape[0] // BATCH_SIZE
	print('batches: {}'.format(batches))
	losses = []
	val_losses = []
	for epoch in range(EPOCHS):
		np.random.shuffle(train_seq)
		
		start = 0
		for i in range(batches):
			seq = train_seq[start: start + BATCH_SIZE]
			start += BATCH_SIZE
			loss = update_model(loss_calculator, seq)

		# 検証する
		val_loss = evaluate(loss_calculator, val_seq)

		# エポック単位の表示		
		average_loss = loss.data / SEQUENCE_SIZE
		average_val_loss = val_loss.data / SEQUENCE_SIZE
		print('epoch:{}, lodd:{}, val_loss:{}'.format(epoch, average_loss, average_val_loss))
		
		losses.append(average_loss)
		val_losses.append(average_val_loss)

	# 保存する
	serializers.save_npz('./chainer_mynet.npz', mynet)
	_pickle.dump(losses, open('./chainer_losss.pkl', 'wb'))
	_pickle.dump(val_losses, open('./chainer_val_losses.pkl', 'wb'))
