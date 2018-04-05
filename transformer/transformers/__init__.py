# from DT_Transformer import DT_Transformer
import importlib
from os import listdir
from os.path import isfile, join, realpath, dirname

mypath = realpath(__file__)
directory = dirname(mypath)
onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
array = []
for file in onlyfiles:
    if file.split('.')[1] == 'py' and file.split('.')[0] != '__init__':
        array.append(file.split('.')[0])
prefix = 'transformers.'

for transformer in array:
    mod = prefix + transformer
    importlib.import_module(mod)
