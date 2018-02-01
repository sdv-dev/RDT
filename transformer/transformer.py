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

class Transformer(object):
	""" This class is responsible for formatting the input table in a way that is machine learning
	friendly
	"""

	def __init__(self, meta_file, table_name):
		""" initialize preprocessor """
		self.meta_file = meta_file
		self.table_name = table_name
		self.params = None

	def process(self):
		""" Returns the processed table """
		raise NotImplementedError

	def transform(self, params):
		""" Does the required transformations to the data """
		raise NotImplementedError

	def reverse_transform(self, params):
		""" Converts data back into original format """
		raise NotImplementedError