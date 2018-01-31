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

	def __init__(self, meta_file, transformers_list):
		""" initialize preprocessor """

		# load json file
		self.meta_file = meta_file
		with open(meta_file, 'r') as f:
			self.meta = json.load(f)
		self.tables = {}
		self.transformers_list = transformers_list
		self.type_map = {}
		for table in self.meta['tables']:
			# get each table
			if table['use']:
				self.type_map[table['name']] = self.get_types(table)
				prefix = op.dirname(meta_file)
				relative_path = op.join(prefix, self.meta['path'], table['path'])
				data_table = pd.read_csv(relative_path)
				self.tables[table['name']] = data_table
		print(self.type_map)

	def hyper_process(self):
		""" Returns the processed table """
		res = []
		for table in self.tables:
			for col_name in list(self.tables[table]):
				for trans in self.transformers_list:
					transformer = trans(self.meta_file, table)
					if transformer.type == self.type_map[table][col_name]:
						res.append(transformer.process(col_name, self.tables[table]))
		return res

	def hyper_transform(self):
		""" Does the required transformations to the data """
		pass

	def hyper_reverse_transform(self):
		""" Converts data back into original format """
		pass

	def get_types(self, table):
		""" Maps every field name to a type """
		res = {}
		for field in table['fields']:
			res[field['name']] = field['type']
		return res
