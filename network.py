#!/usr/bin/python3

import pynn as nn
from testutil import Case


with Case('Network'):
	net = nn.Network([nn.Site(4)], [nn.Site(2)])
	net.add(0, nn.Matrix(4, 2))
	net.add(1, nn.Bias(2))
	net.connect(0, 1)

with Case('Network._Trace'):
	trace = net.newTrace()

print(trace.nodes)
