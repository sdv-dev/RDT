# from DT_Transformer import DT_Transformer
import importlib
l = ['DT_Transformer']
prefix = 'transformers.'

for transformer in l:
	mod = prefix + transformer
	importlib.import_module(mod)
	