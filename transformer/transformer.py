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

	def __init__(self):
		""" initialize preprocessor """
		pass

	def process(self, data, meta):
		""" Returns the processed table """
		raise NotImplementedError

	def transform(self, data, meta):
		""" Does the required transformations to the data """
		raise NotImplementedError

	def reverse_transform(self, data, meta):
		""" Converts data back into original format """
		raise NotImplementedError