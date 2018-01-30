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

class Transformer():
	""" This class is responsible for formatting the input table in a way that is machine learning
	friendly
	"""

	def __init__(self):
		""" initialize preprocessor """
		self.params = None

	def process(self):
		""" Returns the processed table """
		pass

	def transform(self, params):
		""" Does the required transformations to the data """
		pass

	def reverse_transform(self, params):
		""" Converts data back into original format """
		pass