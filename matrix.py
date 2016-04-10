#!/usr/bin/python3

import numpy as np
from pynn import Matrix
import pynn.array as array
from testutil import Case

with Case('Matrix'):
	m = Matrix(3, 2)
	assert(m.inum == 1 and m.onum == 1)
	assert(m.isize == 3 and m.osize == 2)

with Case('Matrix._State'):
	factory = array.newFactory()
	st = m.newState(factory)
	w = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3).T
	st.data.set(w)

with Case('Matrix._State._Gradient'):
	grad = st.newGradient(factory)

with Case('Matrix._State._Rate'):
	rate = st.newRate(factory, 1e-2)
	adagrad = st.newRate(factory, 1e-2, adagrad=True)

with Case('Matrix._Trace'):
	tr = m.newTrace(factory)

with Case('Matrix._Context'):
	src, dst = factory.empty(m.isize), factory.empty(m.osize)
	ctx = m.newContext(factory)
	ctx.state = st
	ctx.trace = tr
	ctx.grad = grad
	ctx.rate = rate
	ctx.src = src
	ctx.dst = dst

with Case('Matrix.transmit'):
	inp = np.array([1, 2, 3])
	src.set(inp)
	m.transmit(ctx)
	assert(np.sum((dst.get() - np.dot(inp, w))**2) < 1e-8)

with Case('Matrix.backprop'):
	outp = np.array([1, 2])
	dst.set(outp)
	m.backprop(ctx)
	assert(np.sum((src.get() - np.dot(w, outp))**2) < 1e-8)
	assert(np.sum((grad.data.get() - np.outer(inp, outp))**2) < 1e-8)

with Case('Matrix._State.learn'):
	st.learn(grad, rate)
	assert(np.sum((st.data.get() - (w - rate.factor*grad.data.get()))**2) < 1e-8)
	adagrad.update(grad)
	st.learn(grad, adagrad)
