#!/usr/bin/python3

import numpy as np
from random import shuffle
import pynn as nn
import pynnui as ui


data = open('20k.txt').read()
words = []
for word in data.split('\n'):
	words.append(word + '\n')

maxlen = max([len(w) for w in words])

chars = sorted(list(set(data)))
ci = {c: i for i, c in enumerate(chars)}
ic = {i: c for i, c in enumerate(chars)}

size = len(chars)

# print(chars)

bsize = 20
shid = 200

factory = nn.array.newFactory()

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

# open('net.svg', 'w').write(ui.Graph(net).svg())

onet = nn.Network(size, size)
onet.addnodes(net)
onet.setinputs(0)
onet.setoutputs(0)
onet.prepare()

state = onet.newState(factory)

nmap = {
	'Wxh': 0,
	'Whh': 1,
	'bh': 3,
	'Why': 6,
	'by': 7
}

save = np.load('state/wordgen.npz')

for key in save:
	state.nodes[0].nodes[nmap[key]].data.set(save[key])


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


class Word:
	def __init__(self, word):
		self.word = word

	def __getitem__(self, i):
		return Entry(ci[self.word[i]], ci[self.word[i + 1]])

	def __len__(self):
		return len(self.word) - 1


def BGen(data, bsize):
	diter = iter(data)
	while True:
		batch = []
		try:
			for _ in range(bsize):
				batch.append(Word(next(diter)))
		except StopIteration:
			pass
		if len(batch) == 0:
			break
		yield batch


def EGen(data):
	while True:
		yield BGen(data, 20)
		shuffle(data)

egen = EGen(words)

lstat = nn.Profiler()


class Callback:
	def __init__(self):
		self.counter = 0

	def __call__(self, ctx):
		print(ctx.loss)
		self.counter += 1
		if self.counter == 20:
			raise StopIteration

teacher = nn.Teacher(
	factory, egen, onet, state,
	adagrad=True, rate=1e-1, clip=5,
	maxlen=maxlen, bmon=Callback()
)

with lstat:
	teacher.teach()

save = np.load('state/wordgen_batch_20_adagrad.npz')
print('weight diff:')
for key in save:
	print(np.sum((state.nodes[0].nodes[nmap[key]].data.get() - save[key])**2))

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

print('total: %f ms' % (1e3*lstat.time))

stats = nn.array.stats
times = [v.time for v in stats.values()]
for v, k in reversed(sorted(zip(times, stats.keys()))):
	if v > 0.:
		print('%f ms: %s' % (1e3*v, k))
