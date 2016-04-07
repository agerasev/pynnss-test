#!/usr/bin/python3

import numpy as np
import pynn.array as array
from testutil import Case


with Case('_FactoryCPU'):
	factory = array.newFactory(dtype=np.int32)

with Case('_FactoryCPU.empty'):
	a = factory.empty((4, 5))
	assert(a.np.shape == (4, 5))

with Case('_FactoryCPU.zeros'):
	a = factory.zeros((4, 5))
	assert(np.sum(a.get()**2) == 0)

with Case('_FactoryCPU.copynp'):
	npa = np.array([1, 2, 3, 4])
	a = factory.copynp(npa)
	assert(a.np[0] == 1 and a.np[3] == 4)
	assert(a.np is not npa)

with Case('_FactoryCPU.copy'):
	a = factory.copynp(np.array([1, 2, 3]))
	b = factory.copy(a)
	assert(np.all(b.np == a.np))
	assert(a.np is not b.np)

with Case('copy'):
	a, b = factory.empty(4), factory.empty(4)
	a.np[0] = 1
	array.copy(b, a)
	a.np[0] = 2
	assert(b.np[0] == 1)

# to be continued ...
