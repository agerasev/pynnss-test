#!/usr/bin/python3

import pynn as nn
from random import shuffle
import numpy as np
import signal


data = open('20k.txt').read()
words = []
for word in data.split('\n'):
	words.append(word + '\n')

max_depth = max([len(w) for w in words])

chars = sorted(list(set(data)))
ci = {c: i for i, c in enumerate(chars)}
ic = {i: c for i, c in enumerate(chars)}

size = len(chars)

# print(chars)

factory = nn.array.newFactory()


class Entry:
	def __init__(self, ichar, ochar):
		self.ichar = ichar
		self.ochar = ochar

	def getinput(self, buf):
		nn.array.clear(buf)
		buf.np[self.ichar] = 1.

	def getouptut(self, buf):
		nn.array.clear(buf)
		buf.np[self.ochar] = 1.


class Series:
	def __init__(self, word):
		self.word = word

	def __getitem__(self, i):
		return Entry(ci[self.word[i]], ci[self.word[i + 1]])

	def __len__(self):
		return len(word) - 1


def batch_gen():
	for word in words:
		yield Series(word)


shid = 200
batch_size = 20
rate_factor = 1e-1

batch = nn.Batch(factory, batch_size, maxlen=max_depth)

opt = {'prof': True}

net = nn.Network(size, size, **opt)

net.addnodes([
	nn.Matrix(size, shid, **opt),
	nn.Matrix(shid, shid, **opt),
	nn.Join(shid, **opt),
	nn.Bias(shid, **opt),
	nn.Tanh(shid, **opt),
	nn.Fork(shid, **opt),
	nn.Matrix(shid, size, **opt),
	nn.Bias(size, **opt),
	nn.SoftmaxLoss(size, **opt)
])

net.connect([
	nn.Path(0, (2, 0)),
	nn.Path(1, (2, 1)),

	nn.Path(2, 3),
	nn.Path(3, 4),
	nn.Path(4, 5),

	nn.Path((5, 1), 1, mem=True),
	nn.Path((5, 0), 6),
	nn.Path(6, 7),
	nn.Path(7, 8)
])

net.setinputs(0)
net.setoutputs(8)

net.prepare()

context = net.newContext(factory)
context.state = state = net.newState(factory)

nmap = {
	'Wxh': 0,
	'Whh': 1,
	'bh': 3,
	'Why': 6,
	'by': 7
}

save = np.load('state/wordgen.npz')

for key in save:
	state.nodes[nmap[key]].data.set(save[key])

context.trace = net.newTrace(factory)
context.grad = state.newGradient(factory)
context.rate = state.newRate(factory, rate_factor, adagrad=True)

b = 0
p = 0
smooth_loss = 0
losses = []
epoch = 0
epochs = []

done = False


def signal_handler(signal, frame):
	global done
	done = True
signal.signal(signal.SIGINT, signal_handler)

lstat = nn.Profiler()

counter = 0
show_period = 20
gen = batch_gen()
while not done:
	if p >= len(words):
		p = 0
		b = 0
		epoch += 1
	if p == 0:
		# shuffle(words)
		pass

	loss = batch.do(net, context, gen)
	with lstat:
		context.grad.clip(5e0)
		context.rate.update(context.grad)
		context.state.learn(context.grad, context.rate)

	counter += 1
	if counter == 20:
		save = np.load('state/wordgen_batch_20_adagrad.npz')
		for key in save:
			print(np.sum((state.nodes[nmap[key]].data.get() - save[key])**2))

		print('fnet: %f' % (1e3*net.fstat.time))
		tac = 0.
		for n in net.nodes:
			tac += n.fstat.time
		print(' fnodes: %f ms' % (1e3*tac))

		print('bnet: %f ms' % (1e3*net.bstat.time))
		tac = 0.
		for n in net.nodes:
			tac += n.bstat.time
		print(' bnodes: %f ms' % (1e3*tac))

		print('learn: %f ms' % (1e3*lstat.time))

		stats = nn.array.stats
		times = [v.time for v in stats.values()]
		for v, k in reversed(sorted(zip(times, stats.keys()))):
			if v > 0.:
				print('%f ms: %s' % (1e3*v, k))

		exit()

	smooth_loss = 0.9*smooth_loss + 0.1*loss

	if (b+1) % show_period == 0:
		smooth_epoch = epoch + p/len(words)
		print('%f: %f' % (smooth_epoch, smooth_loss))
		# break

	p += batch_size
	b += 1

print('done')

'''
alphabet = 'abcdefghijklmnopqrstuvwxyz'

for j in range(len(alphabet)):

	state = net.newState()

	a = ci[alphabet[j]]
	print(alphabet[j], end='')

	for i in range(0x40):
		lin = [0]*size
		lin[a] = 1
		vins = [np.array(lin)]
		vouts = net.transmit(state, vins)
		prob = np.exp(vouts[0])/np.sum(np.exp(vouts[0]))
		a = np.random.choice(range(size), p=prob)
		letter = ic[a]

		if letter == '\n':
			break
		print(letter, end='')
	print()
'''
