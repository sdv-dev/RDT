# from DT_Transformer import DT_Transformer
import importlib
from os import listdir
from os.path import isfile, join, realpath, dirname

mypath = realpath(__file__)
directory = dirname(mypath)
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
l = []
for file in onlyfiles:
	if file.split('.')[1] == 'py' and file.split('.')[0] != '__init__':
		l.append(file.split('.')[0])
# l = ['DT_Transformer']
prefix = 'transformers.'

for transformer in l:
	mod = prefix + transformer
	importlib.import_module(mod)
	