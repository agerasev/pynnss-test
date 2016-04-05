#!/usr/bin/python3

import numpy as np
from pynn import Matrix
import pynn.array as arr
from testutil import Case


with Case('Matrix'):
	m = Matrix(3, 2)
	assert(m.inum == 1 and m.onum == 1)
	assert(m.isize == 3 and m.osize == 2)

with Case('Matrix._State'):
	st = m.newState()
	w = np.array([1, 2, 3, 4, 5, 6]).reshape(2, 3)
	st.data.set(w)

with Case('Matrix._State._Gradient'):
	grad = st.newGradient()

with Case('Matrix._State._Rate'):
	rate = st.newRate(1e-2)
	adagrad = st.newRate(1e-2, adagrad=True)

with Case('Matrix._Memory'):
	mem = m.newMemory()

with Case('Matrix._Error'):
	err = m.newError()

with Case('Matrix._Context'):
	src, dst = arr.Array(m.isize), arr.Array(m.osize)
	comp = {
		'state': st,
		'mem': mem,
		'grad': grad,
		'err': err,
		'rate': rate
	}
	ctx = m.newContext(src, dst, **comp)

with Case('Matrix.transmit'):
	inp = np.array([1, 2, 3])
	src.set(inp)
	m.transmit(ctx)
	assert(np.sum((dst.get() - np.dot(w, inp))**2) < 1e-8)

with Case('Matrix.backprop'):
	outp = np.array([1, 2])
	dst.set(outp)
	m.backprop(ctx)
	assert(np.sum((src.get() - np.dot(outp, w))**2) < 1e-8)
	assert(np.sum((grad.data.get() - np.outer(outp, inp))**2) < 1e-8)

with Case('Matrix._State.learn'):
	st.learn(grad, rate)
	assert(np.sum((st.data.get() - (w - rate.factor*grad.data.get()))**2) < 1e-8)
	adagrad.update(grad.data)
	st.learn(grad, adagrad)
