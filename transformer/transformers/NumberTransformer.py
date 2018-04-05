import exrex
import itertools
import numpy as np
import pandas as pd
import time
import pdb
import json
import os.path as op
import sys

from BaseTransformer import *

class NumberTransformer(BaseTransformer):
	""" 
	This class represents the datetime transformer for SDV 
	"""

	def __init__(self):
		""" initialize transformer """
		super(NumberTransformer, self).__init__()
		self.type = 'number'

	def fit_transform(self, col, col_meta):
		""" Returns a tuple (transformed_table, new_table_meta) """
		out = pd.DataFrame(columns=[])
		col_name = col_meta['name']
		print(col_meta)
		subtype = col_meta['subtype']
		if subtype == 'integer':
			out[col_name] = col.apply(self.get_val)
		# replace missing values
		# create an extra column for missing values if they exist in the data
		new_name = '?' + col_name
		# if are just processing child rows, then the name is already known
		out[new_name] = pd.notnull(col) * 1
		return out

	def transform(self, col, col_meta):
		""" Does the required transformations to the data """
		return self.fit_transform(col, col_meta)

	def reverse_transform(self, col, col_meta):
		""" Converts data back into original format """
		output = pd.DataFrame(columns=[])
		subtype = col_meta['subtype']
		col_name = col_meta['name']
		fn = self.get_number_converter(col_name, subtype)
		data = col.to_frame()
		output[col_name] = data.apply(fn, axis=1)
		return output
			
	def get_val(self, x):

		try:
			return int(round(x))
		except (ValueError, TypeError):
			return np.nan

	def get_number_converter(self, col, meta):
		'''Returns a converter that takes in a value and turns it into an
		   integer, if necessary

		:param col: name of column
		:type col: str
		:param missing: true if column has NULL values
		:type missing: bool
		:param meta: type of column values
		:type meta: str

		:returns: function
		'''

		def safe_round(x):
			if meta == 'integer':
				if x[col] == 'NaN' or np.isnan(x[col]):
					return np.nan
				return int(round(x[col]))
			return x[col]

		return safe_round