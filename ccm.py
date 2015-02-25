#!/usr/bin/env python

import argparse

import os
import sys
import numpy as np
import scipy.spatial.distance
import sqlite3
import matplotlib.pyplot as plt
from pyembedding import *

def parse_variable_pairs(col_names, variable_pairs_str):
	var_pairs = list()
	if variable_pairs_str is None:
		for v1 in col_names:
			for v2 in col_names:
				var_pairs.append((v1, v2))
	else:
		pair_strs = [x.strip() for x in variable_pairs_str.split(',')]
		for pair_str in pair_strs:
			var_names = [x.strip() for x in pair_str.split(':')]
			if len(var_names) != 2:
				raise Exception('{0} does not specify two variables separated by :.'.format(pair_str))
			for var_name in var_names:
				if not var_name in col_names:
					raise Exception('{0} does match a column in the database.'.format(var_name))
			var_pairs.append(tuple(var_names))
	
	return var_pairs

def parse_col_names(col_names_str):
	return [x.strip() for x in col_names_str.split(',')]

if __name__ == '__main__':
	# Construct arguments
	parser = argparse.ArgumentParser(
		description='Run a convergent cross-mapping (CCM) analysis.',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument(
		'input_filename', metavar='<input-file>', type=str, help='Input filename (SQLite format).'
	)
	parser.add_argument(
		'input_table_name', metavar='<input-table-name>', type=str, help='Name of data table in input file.'
	)
	parser.add_argument(
		'output_filename', metavar='<output-file>', type=str,
		help='Output filename (SQLite format).'
	)
	parser.add_argument(
		'--variable-pairs', '-V', metavar='<cause>:<effect>,<cause>:<effect>...', type=str,
		help='Pairs of variables to test for causality; cause and effect are separated by a colon (:) and pairs are separated by a comma (,).' +
		'If omitted, all pairs of variables corresponding to loaded table columns will be tested.'
	)
	parser.add_argument(
		'--library-sizes', '-L', metavar='<start-L>:<end-L>:<skip>|<L1,L2,...>', type=str,
		help='Sizes of prediction libraries to test. ' + 
			'<start-L>:<end-L>:<skip> will use <start-L>, <start-L> + <skip>, ..., <end-L>. ' + 
			'<end-L> will be included if it is equal to <start-L> + <skip>*N for some N. ' + 
			'If omitted, only the minimum and maximum possible library sizes will be used.'
	)
	parser.add_argument(
		'--replicates', '-R', metavar='<n-reps>', type=int, default=100,
		help='Number of replicates to run for each library size.'
	)
	parser.add_argument(
		'--embedding-dimension', '-E', metavar='<E>', type=int, default=3,
		help='Embedding dimension used for prediction.'
	)
	parser.add_argument(
		'--tau', '-t', metavar='<tau>', type=int, default=1,
		help='Time lag used to reconstruct attractor.'
	)
	parser.add_argument(
		'--columns', '-C', metavar='<col1>,<col2>,...', type=str,
		help='Columns to load from the SQLite table.'
	)
	parser.add_argument(
		'--filter', '-f', metavar='<filter>', type=str,
		help='SQL filter to select rows (requires SQLite input data). E.g., "time >= 100.0" will use "WHERE time >= 100.0" when reading rows.'
	)
	parser.add_argument(
		'--overwrite-output', '-o', action='store_true',
		help='Overwrite output file if it already exists.'
	)
	args = parser.parse_args()
	
	# Check input & output files
	if args.input_filename == args.output_filename:
		parser.exit('Input filename and output filename cannot be the same.')
	if os.path.exists(args.output_filename):
		if args.overwrite_output:
			os.path.remove(args.output_filename)
		else:
			parser.error('Output filename {0} exists. Delete it or use --overwrite-output.'.format(args.output_filename))
	
	# Load input data
	print args.filter
	try:
		with sqlite3.connect(args.input_filename) as db:
			col_names, columns = read_table(
				db, args.input_table_name, col_names=parse_col_names(args.columns), filter=args.filter
			)
	except Exception as e:
		parser.error(e)
	print col_names
	print columns
	
	# Parse variable pairs
	try:
		var_pairs = parse_variable_pairs(col_names, args.variable_pairs)
	except Exception as e:
		parser.error(e)
	
	output_filename = args.output_filename
	if not output_filename.endswith('.sqlite'):
		output_filename += '.sqlite'
