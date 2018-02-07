import exrex
import itertools
import numpy as np
import pandas as pd
import progressbar
import shelve
import time
import pdb
import json
import os.path as op

from transformers import *
from dateutil import parser

class HyperTransformer:
	""" This class is responsible for formatting the input table in a way that is machine learning
	friendly
	"""

	def __init__(self):
		""" initialize preprocessor """

		# load json file
		# self.meta_file = meta_file
		# with open(meta_file, 'r') as f:
		# 	self.meta = json.load(f)
		# self.tables = {}
		# self.transformers_list = transformers_list
		# self.type_map = {}
		# for table in self.meta['tables']:
		# 	# get each table
		# 	if table['use']:
		# 		self.type_map[table['name']] = self.get_types(table)
		# 		prefix = op.dirname(meta_file)
		# 		relative_path = op.join(prefix, self.meta['path'], table['path'])
		# 		data_table = pd.read_csv(relative_path)
		# 		self.tables[table['name']] = data_table
		# print(self.type_map)
		self.transformers = []

	def get_class(self, class_name):
		return getattr(globals()[class_name], class_name)

	def hyper_fit_transform(self, transformers_list, table, table_meta):
		""" Returns the processed table after going through each transform 
		and adds fitted transformers to the hyper class
		"""
		# res = []
		# for table in self.tables:
		# 	for col_name in list(self.tables[table]):
		# 		for trans in self.transformers_list:
		# 			transformer = trans(self.meta_file, table)
		# 			if transformer.type == self.type_map[table][col_name]:
		# 				res.append(transformer.process(col_name, self.tables[table]))
		# return res
		data = table
		for transformer in transformers_list:
			trans_class = self.get_class(transformer)
			t = trans_class()
			data = t.fit_transform(data, table_meta)
			self.transformers.append(t)
		return data

	def hyper_transform(self, table, table_meta):
		""" Does the required transformations to the table """
		# res = []
		# for table in self.tables:
		# 	for col_name in list(self.tables[table]):
		# 		for trans in self.transformers_list:
		# 			transformer = trans(self.meta_file, table)
		# 			if transformer.type == self.type_map[table][col_name]:
		# 				res.append(transformer.process(col_name, self.tables[table]))
		# return res
		data = table
		for transformer in self.transformers:
			data = transformer.transform(data, table_meta)
		return data

	def hyper_reverse_transform(self, table, table_meta):
		""" Converts data back into original format by looping
		over all transformers and doing the reverse
		"""
		# res = []
		# for table in self.tables:
		# 	for col_name in list(self.tables[table]):
		# 		for trans in self.transformers_list:
		# 			transformer = trans(self.meta_file, table)
		# 			if transformer.type == self.type_map[table][col_name]:
		# 				res.append(transformer.process(col_name, self.tables[table]))
		# return res
		data = table
		for transformer in self.transformers:
			data = transformer.reverse_transform(data, table_meta)
		return data

	def get_types(self, table):
		""" Maps every field name to a type """
		res = {}
		for field in table['fields']:
			res[field['name']] = field['type']
		return res

if __name__ == "__main__":
	meta_file = '../data/Airbnb_demo_meta.json'
	with open(meta_file, 'r') as f:
		meta = json.load(f)
	tables = {}
	type_map = {}
	for table in meta['tables']:
		# get each table
		if table['use']:
			prefix = op.dirname(meta_file)
			relative_path = op.join(prefix, meta['path'], table['path'])
			data_table = pd.read_csv(relative_path)
			tables[table['name']] = (data_table, table)

	# test out hyper_transformer
	ht_map = {}
	tl = ['DT_Transformer']
	transformed = {}
	for table_name in tables:
		table, table_meta = tables[table_name]
		ht = HyperTransformer()
		transformed[table_name] = ht.hyper_fit_transform(tl, table, table_meta)
		ht_map[table_name] = ht
	for key in transformed:
		ht = ht_map[key]
		table = transformed[key]
		# print(table)
		table_meta = tables[key][1]
		print(ht.hyper_reverse_transform(table, table_meta))
