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
from dateutil import parser

class DT_Transformer(BaseTransformer):
	""" 
	This class represents the datetime transformer for SDV 
	"""

	def __init__(self):
		""" initialize transformer """
		super(DT_Transformer, self).__init__()
		self.type = 'datetime'

	def fit_transform(self, table, table_meta):
		""" Returns a tuple (transformed_table, new_table_meta) """
		out = pd.DataFrame(columns=[])
		for field in table_meta['fields']:
			col_name = field['name']
			if field['type'] == self.type:
				# convert datetime values
				col = table[col_name]
				out[col_name] = col.apply(self.get_val)

				# replace missing values
				# create an extra column for missing values if they exist in the data
				new_name = '?' + col_name
				# if are just processing child rows, then the name is already known
				if new_name in list(table):
					out[new_name] = pd.notnull(col) * 1
				# if this is our first time processing child rows, then need to see
				# if this should be broken down
				elif pd.isnull(col).any() and not pd.isnull(col).all():
					out[new_name] = pd.notnull(col) * 1
			else:
				out[col_name] = table[col_name] 
		return out

	def transform(self, table, table_meta):
		""" Does the required transformations to the data """
		return self.process(table, table_meta)

	def reverse_transform(self, table, table_meta):
		""" Converts data back into original format """
		output = pd.DataFrame(columns=[])
		for field in table_meta['fields']:
			col_name = field['name']
			if field['type'] == self.type:
				date_format = field['format']
				fn = self.get_date_converter(col_name, date_format)
				data = table[col_name].to_frame()
				output[col_name] = data.apply(fn, axis=1)
			else:
				output[col_name] = table[col_name]
		return output

	def get_val(self, x):
		""" Converts datetime to number """
		try:
			tmp = parser.parse(x).timetuple()
			return time.mktime(tmp)*1e9
		except (ValueError, AttributeError, TypeError):
			# if we return pd.NaT, pandas will exclude the column
			# when calculating covariance, so just use np.nan
			return np.nan

	def get_date_converter(self, col, meta):
		'''Returns a converter that takes in an integer representing ms
		   and turns it into a string date

		:param col: name of column
		:type col: str
		:param missing: true if column has NULL values
		:type missing: bool
		:param meta: type of column values
		:type meta: str

		:returns: function
		'''

		def safe_date(x):
			if '?' + col in x:
				missing = True
			else:
				missing = False

			if missing and x['?' + col] == 0:
				return np.nan
			# TODO: Figure out right way to check missing
			t = x[col]
			if np.isnan(t):
				return np.nan
			tmp = time.gmtime(float(t)/1e9)
			return time.strftime(meta, tmp)

		return safe_date