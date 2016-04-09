#!/usr/bin/python3

import pynn as nn
from random import shuffle
import numpy as np
import signal


data = open('20k.txt').read()
words = []
for word in data.split('\n'):
	words.append(word + '\n')

chars = sorted(list(set(data)))
ci = {c: i for i, c in enumerate(chars)}
ic = {i: c for i, c in enumerate(chars)}

size = len(chars)

# print(chars)

factory = nn.array.newFactory()

max_depth = 64


def do_batch(net, ctx, pos, batch_size):
	loss = 0
	ctx.grad.clear()
	src = factory.empty(size)
	dst = factory.empty(size)

	ctx.src = src
	ctx.dst = dst

	imem = ctx.state.newMemory(factory)
	ierr = ctx.state.newError(factory)
	trace_stack = [net.newTrace(factory) for i in range(max_depth)]
	out_stack = [factory.empty(size) for i in range(max_depth)]

	for i in range(min(batch_size, len(words) - pos)):
		word = words[pos + i]
		depth = len(word)

		imem.copyto(ctx.mem)  # TODO: use set(other) instead

		for l in range(depth - 1):
			a = ci[word[l]]
			lin = [0.]*size
			lin[a] = 1.
			src.set(np.array(lin))

			# feedforward
			net.transmit(ctx)
			ctx.trace.copyto(trace_stack[l])
			nn.array.copy(out_stack[l], dst)

		ierr.copyto(ctx.err)

		for l in reversed(range(depth - 1)):
			a = ci[word[l + 1]]
			lres = [0.]*size
			lres[a] = 1.
			vres = np.array(lres)
			vin = out_stack[l].get()
			vout = np.exp(vin)/np.sum(np.exp(vin))  # softmax
			dst.set(vout - vres)
			loss += -np.log(vout[a])

			# backpropagate
			trace_stack[l].copyto(ctx.trace)
			net.backprop(ctx)

	# if i == 1 and l == depth - 2 - 0:
	# print(ic[a])
	# data = ctx.grad.nodes[0].data.get()
	# print(data)
	# print(np.sum((data - np.load('state/gradNM0.npy'))**2))
	# exit()

	ctx.grad.mul(1/batch_size)
	return loss/batch_size

shid = 200
batch_size = 20
rate_factor = 1e-1

net = nn.Network(size, size)

net.addnodes([
	nn.Matrix(size, shid),
	nn.Matrix(shid, shid),
	nn.Join(shid),
	nn.Bias(shid),
	nn.Tanh(shid),
	nn.Fork(shid),
	nn.Matrix(shid, size),
	nn.Bias(size)
])

net.connect([
	nn.Path(0, (2, 0)),
	nn.Path(1, (2, 1)),

	nn.Path(2, 3),
	nn.Path(3, 4),
	nn.Path(4, 5),

	nn.Path((5, 1), 1, mem=True),
	nn.Path((5, 0), 6),
	nn.Path(6, 7)
])

net.order = [0, 1, 2, 3, 4, 5, 6, 7]

net.setinputs(0)
net.setoutputs(7)

context = net.newContext(factory)
context.state = state = net.newState(factory)

save = np.load('state/wordgen.npz')
state.nodes[0].data.set(save['Wxh'])
state.nodes[1].data.set(save['Whh'])
state.nodes[3].data.set(save['bh'])
state.nodes[6].data.set(save['Why'])
state.nodes[7].data.set(save['by'])

context.trace = net.newTrace(factory)
context.grad = state.newGradient(factory)
context.rate = state.newRate(factory, rate_factor)
context.mem = state.newMemory(factory)
context.err = state.newError(factory)

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

show_period = 20
while not done:
	if p >= len(words):
		p = 0
		b = 0
		epoch += 1
	if p == 0:
		# shuffle(words)
		pass

	loss = do_batch(net, context, p, batch_size)
	context.grad.clip(5e0)
	# context.rate.update(context.grad)
	context.state.learn(context.grad, context.rate)

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
