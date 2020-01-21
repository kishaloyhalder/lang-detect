class Logger():
	def __init__(self, filepath):
		self._path = filepath
		f = open(self._path, 'w')
		f.close()
	def log(self, st):
		f = open(self._path, 'a')
		f.write(st)
		f.write('\n')
		f.close()