#!/usr/bin/python3


import pynn as nn
from testutil import Case


with Case('Network'):
	net = nn.Network(4, 2)
	net.addnodes([nn.Matrix(4, 2), nn.Bias(2), nn.Tanh(2)])
	net.connect([(0, 1), (1, 2)])
	net.setinputs(0)
	net.setoutputs(2)

factory = nn.array.newFactory()

with Case('Network._Trace'):
	trace = net.newTrace(factory)

with Case('Network._State'):
	state = net.newState(factory)

with Case('Network._State._Memory'):
	mem = state.newMemory(factory)

with Case('Network._State._Error'):
	err = state.newError(factory)

with Case('Network._State._Gradient'):
	grad = state.newGradient(factory)

with Case('Network._State._Rate'):
	rate = state.newRate(factory, 1e-2, adagrad=True)

with Case('Network._Context'):
	ctx = net.newContext(factory)
