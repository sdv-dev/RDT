import exrex
import itertools
import numpy as np
import pandas as pd
import time
import pdb
import json
import os.path as op
import sys

sys.path.append( op.dirname( op.dirname( op.abspath(__file__) ) ) )
from transformer import *
from dateutil import parser

class DT_Transformer(Transformer):
	""" This class is responsible for formatting the input table in a way that is machine learning
	friendly
	"""

	def __init__(self, meta_file, table_name):
		""" initialize transformer """
		super(DT_Transformer, self).__init__(meta_file, table_name)
		self.type = 'datetime'
		# get meta info for table
		with open(meta_file, 'r') as f:
			meta = json.load(f)
		for table in meta['tables']:
			if table['name'] == table_name:
				self.table_meta = table

	def process(self, col_name, data):
		""" Returns the processed table """

		# convert datetime values
		col = data[col_name]
		out = pd.DataFrame(columns=[])
		params = []
		out[col_name] = col.apply(self.get_val)

		# replace missing values
		# create an extra column for missing values if they exist in the data
		new_name = '?' + col_name
		# if are just processing child rows, then the name is already known
		if new_name in list(data):
			out[new_name] = pd.notnull(col) * 1
		# if this is our first time processing child rows, then need to see
		# if this should be broken down
		elif pd.isnull(col).any() and not pd.isnull(col).all():
			out[new_name] = pd.notnull(col) * 1
			params.append(('categorical',''))
		return (out, params)

	def transform(self, col_name, data, params):
		""" Does the required transformations to the data """
		return self.process(col_name, data)

	def reverse_transform(self, col_name, data, params, output=None):
		""" Converts data back into original format """
		if output == None:
			output = pd.DataFrame(columns=[])
		for field in self.table_meta['fields']:
			if field['name'] == col_name:
				date_format = field['format']
		fn = self.get_date_converter(col_name, date_format)
		output[col_name] = data.apply(fn, axis=1)
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

			t = x[col]
			tmp = time.gmtime(float(t)/1e9)
			return time.strftime(meta, tmp)

		return safe_date