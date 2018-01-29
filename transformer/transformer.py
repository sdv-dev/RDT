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

	def __init__(self):
		""" initialize preprocessor """
		self.params = None
		super().__init__()

	@abstractmethod
	def process(self):
		""" Returns the processed table """
		pass

	@abstractmethod
	def transform(self, params):
		""" Does the required transformations to the data """
		pass

	@abstractmethod
	def reverse_transform(self, params):
		""" Converts data back into original format """
		pass