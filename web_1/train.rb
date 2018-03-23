require 'pycall/import'
include PyCall::Import

pyimport 'chainer'
pyimport 'numpy', as: :np
pyfrom 'chainer', import: :training
pyfrom 'chainer.training', import: :extensions
pyimport 'random'
pyimport 'collections'
PyCall.sys.path.append(__dir__)
pyfrom 'data', import: :NData
pyfrom 'model', import: :Model


	data = NData.load_data("on.txt", "off.txt")
	random.seed(1)
	np.random.seed(1)
	random.shuffle(data)
	dataset = NData.make_dataset(data)
	epoch = 200
	batchsize = 100
	units = data[0][0].to_a.size
	m = Model.(units)
	model = m.get_model()
	
	optimizer = chainer.optimizers.Adam().new
	optimizer.setup(model)
	test_ratio = 0.05
	nt = (data.to_a.size*test_ratio).to_i
	test = dataset[0...nt]
	train = dataset[nt..-1]
	train_iter = chainer.iterators.SerialIterator.(train, batchsize)
	test_iter = chainer.iterators.SerialIterator.(test, batchsize, repeat=false, shuffle=false)
	updater = training.StandardUpdater.(train_iter, optimizer)
	trainer = training.Trainer.(updater, [epoch, 'epoch'], out='result')
#	trainer.append.(extensions.Evaluator(test_iter, model))
#	trainer.append(extensions.dump_graph('main/loss'))
#	trainer.append(extensions.snapshot(), trigger=[epoch, 'epoch'])
#	trainer.extend(extensions.LogReport())
#	trainer.extend(extensions.PrintReport(
#				['epoch', 'main/loss', 'validation/main/loss',
#				'main/accuracy', 'validation/main/accuracy']))
#	trainer.extend(extensions.ProgressBar())
	
	trainer.run()
	m.save('test.model')
