import pynnorig as nn
from random import shuffle
from copy import copy
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


def do_batch(net, pos, batch_size):
	loss = 0
	grad = net.newGradient()

	for i in range(min(batch_size, len(words) - pos)):
		word = words[pos + i]
		depth = len(word)

		state = net.newState()
		state_stack = []
		vouts_stack = []

		for l in range(depth - 1):
			a = ci[word[l]]
			lin = [0.]*size
			lin[a] = 1.
			vins = [np.array(lin)]

			# feedforward
			state = copy(state)
			vouts = net.transmit(state, vins)
			state_stack.append(state)
			vouts_stack.append(vouts)

		error = net.newError()

		for l in range(depth - 1):
			a = ci[word[depth - l - 1]]
			lres = [0.]*size
			lres[a] = 1.
			vres = np.array(lres)
			vin = vouts_stack.pop()[0]
			vout = np.exp(vin)/np.sum(np.exp(vin))  # softmax
			verrs = [vout - vres]
			loss += -np.log(vout[a])

			# backpropagate
			net.backprop(grad, error, state_stack.pop(), verrs)

	# if i == 1 and l == 0:
	# print(ic[a])
	# data = grad.nodes[0].state
	# np.save('state/gradNM0.npy', data)
	# print(data)
	# exit()

	grad.mul(1/batch_size)
	return (grad, loss/batch_size)

shid = 200
batch_size = 20
rate_factor = 1e-1

net = nn.Network(1, 1)

saves = np.load('state/wordgen.npz')

net.nodes[0] = nn.MatrixProduct(size, shid)  # W_xh
net.nodes[0].state = saves['Wxh']
net.nodes[1] = nn.MatrixProduct(shid, shid)  # W_hh
net.nodes[1].state = saves['Whh']
net.nodes[2] = nn.Join(shid, 2)
net.nodes[3] = nn.Bias(shid)
net.nodes[3].state = saves['bh']
net.nodes[4] = nn.Tanh(shid)
net.nodes[5] = nn.Fork(shid, 2)
net.nodes[6] = nn.MatrixProduct(shid, size)  # W_hy
net.nodes[6].state = saves['Why']
net.nodes[7] = nn.Bias(size)
net.nodes[7].state = saves['by']

# np.savez('state/wordgen.npz', **saves)

net.link(nn.Path((-1, 0), (0, 0)))
net.link(nn.Path((0, 0), (2, 0)))
net.link(nn.Path((1, 0), (2, 1)))

net.link(nn.Path((2, 0), (3, 0)))
net.link(nn.Path((3, 0), (4, 0)))
net.link(nn.Path((4, 0), (5, 0)))

net.link(nn.Path((5, 1), (1, 0), np.zeros(shid)))
net.link(nn.Path((5, 0), (6, 0)))
net.link(nn.Path((6, 0), (7, 0)))
net.link(nn.Path((7, 0), (-1, 0)))

rate = nn.RateAdaGrad(net, rate_factor)

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

lprof = nn.Node.Profiler()

counter = 0
show_period = 20
while not done:
	if p >= len(words):
		p = 0
		b = 0
		epoch += 1
	if p == 0:
		pass  # shuffle(words)

	(grad, loss) = do_batch(net, p, batch_size)
	with lprof:
		grad.clip(5e0)
		rate.update(grad)
		net.learn(grad, rate)

	counter += 1
	if counter == 20:
		print('fnet: %f' % (1e3*net.fprof.time))
		tac = 0.
		for n in net.nodes.values():
			tac += n.fprof.time
		print(' fnodes: %f' % (1e3*tac))

		print('bnet: %f' % (1e3*net.bprof.time))
		tac = 0.
		for n in net.nodes.values():
			tac += n.bprof.time
		print(' bnodes: %f' % (1e3*tac))

		print('learn: %f' % (1e3*lprof.time))

		np.savez('state/wordgen_batch_20_adagrad.npz', **{
			'Wxh': net.nodes[0].state,
			'Whh': net.nodes[1].state,
			'bh': net.nodes[3].state,
			'Why': net.nodes[6].state,
			'by': net.nodes[7].state
		})
		exit()

	smooth_loss = 0.9*smooth_loss + 0.1*loss

	if (b+1) % show_period == 0:
		smooth_epoch = epoch + p/len(words)
		print('%f: %f' % (smooth_epoch, smooth_loss))
		# break

	p += batch_size
	b += 1
