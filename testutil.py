#!/usr/bin/python3


class Case:
	def __init__(self, name):
		self.name = name

	def __enter__(self):
		print(self.name + ' ... ', end='')

	def __exit__(self, type, value, traceback):
		if value is not None:
			print('error')
		else:
			print('ok')
