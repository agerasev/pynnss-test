#!/usr/bin/python3

import numpy as np
import pynn.array as arr
from testutil import Case


with Case('Array.__init__'):
	a = arr.Array(np.array([1, 2, 3, 4]), dtype=int)
	assert(a.np[0] == 1 and a.np[3] == 4)

with Case('copy'):
	a, b = arr.Array(4), arr.Array(4)
	a.np[0] = 1
	arr.copy(b, a)
	a.np[0] = 2
	assert(b.np[0] == 1)

# to be continued ...
