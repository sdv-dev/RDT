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

from dateutil import parser

class HyperTransformer:
	""" This class is responsible for formatting the input table in a way that is machine learning
	friendly
	"""

	def __init__(self, meta_file, functions_list):
		""" initialize preprocessor """

		# load json file
		with open(meta_file, 'r') as f:
			self.meta = json.load(f)
		self.tables = []
		self.functions_list = functions_list
		for table in self.meta['tables']:
			# get each table
			if table['use']:
				prefix = op.dirname(meta_file)
				relative_path = op.join(prefix, self.meta['path'], table['path'])
				data_table = pd.read_csv(relative_path)
				self.tables.append(data_table)
		print(self.tables)

	def hyper_process(self):
		""" Returns the processed table """
		pass

	def hyper_transform(self):
		""" Does the required transformations to the data """
		pass

	def hyper_reverse_transform(self):
		""" Converts data back into original format """
		pass