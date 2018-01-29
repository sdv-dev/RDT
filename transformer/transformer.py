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

from abc import ABC, abstractmethod
from dateutil import parser

class Transformer(ABC):
	""" This class is responsible for formatting the input table in a way that is machine learning
	friendly
	"""

	def __init__(self, meta_type):
		""" initialize preprocessor """
		self.type = meta_type
		self.params = None

	def hyper_process(self, table):
		""" Returns the processed table """
		pass

	def hyper_transform(self, table, params):
		""" Does the required transformations to the data """
		pass

	def hyper_reverse_transform(self, table, params):
		""" Converts data back into original format """
		pass