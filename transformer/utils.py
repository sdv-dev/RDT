import json
import pandas as pd
import os.path as op

def get_table_dict(meta_file):
	""" 
	This function parses through a meta file and extracts the tables

	Returns dictionary mapping table name to tuple of (table, table_meta)
	"""
	table_dict = {}
	with open(meta_file, 'r') as f:
		meta = json.load(f)
	for table in meta['tables']:
		if table['use']:
			prefix = op.dirname(meta_file)
			relative_path = op.join(prefix, meta['path'], table['path'])
			data_table = pd.read_csv(relative_path)
			table_dict[table['name']] = (data_table, table)
	return table_dict

