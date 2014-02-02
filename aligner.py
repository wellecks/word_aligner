### Word Aligner - Statistical Machine Translation
### Sean Welleck | 2014
#
# Top level module for performing word alignment.
# Contains generic functions for training and using a Model.
# Contains IO functions for loading and printing data.

from model import Model, IBMM1, BayesM
import sys
from collections import defaultdict

# Trains model parameters with the data. Modifies the state of the model.
# Input:  model - a generic Model object
#         data  - a list of sentence pairs of the form
#         iters - number of iterations (default = 10)
#                [ [french1, ... , frenchm], [eng1, ... , engn] ]
# Output: model - the trained model         
def train_model(model, data, iters=10):
	sys.stderr.write("Training model...\n")
	model.train(data, iters)
	return model

# Given a (trained) model, returns the optimal alignment for each
# sentence pair.
# Input:  model - a generic, trained Model object
#         data  - a list of sentence pairs of the form
#                [ [french1, ... , frenchm], [eng1, ... , engn] ]
# Output: alignments - a list of row alignments of the form
#				 [ [(1,2), (2,1)], ... , [(5,2), (4,5)] ]   
#                where (i,j) denotes foreign word at position i
#                aligned with english word at position j.
def align(model, data):
	sys.stderr.write("Aligning words...\n")
	return model.align(data)

# Read and format input from parallel corpus files.
# Input:  e_fname - filename of English corpus
#         f_fname - filename of Foreign corpus
#         num_sents  - number of sentences to load
# Output: bi_text    - [[f, e]] where f = [foreign words]
#                                     e = [english words]
def load_input(e_fname, f_fname, num_sents, reverse=False):
	if reverse:
		pairs = zip(open(e_fname), open(f_fname))[:num_sents]
	else:
		pairs = zip(open(f_fname), open(e_fname))[:num_sents]
	bitext = [[sentence.strip().split() 
	           for sentence in pair] 
	           for pair in pairs]
	# add a blank word in each french sentence
	for (f, e) in bitext:
		f.append(None)
	return bitext

# Given a list of pairs of alignments, create a table for
# each sentence pair. table[i][j] = 1 if english word i
# is translated to foreign word j, otherwise 0.
# Input:	alignments - alignments as formatted by align()
# Output:	tables     - length n list of tables, each table is
#						 (len(e)) x (len(f)) for a sentence pair (e,f)
def mk_align_tables(alignments):
	tables = []
	for sentence in alignments:
		table = defaultdict(lambda: defaultdict(int))
		for (f_i, e_j) in sentence:
			table[e_j][f_i] = 1
		tables.append(table)
	return tables

# Converts a list of alignment tables into a list of alignment pairs.
# Input:	tables - list of table as formatted by mk_align_tables()
# Output: 	alignments - alignments as formatted by align()
def tables_to_aligns(tables):
	alignments = []
	for (n, table) in enumerate(tables):
		row_alignments = []
		for e_i in table.keys():
			for f_j in table[e_i].keys():
				if table[e_i][f_j] == 1:
					row_alignments.append((f_j, e_i))
		alignments.append(row_alignments)
	return alignments

# Return the intersection of two lists of alignment tables.
# Input:	ts1 - list of alignments as formatted by mk_align_tables()
#			ts2 - list of alignments as formatted by mk_align_tables()
# Output:	ts_int - intersection of tables from ts1 and ts2 
def tables_intersect(ts1, ts2):
	ts_int = []
	for (n, (t1, t2)) in enumerate(zip(ts1, ts2)):
		t_int = defaultdict(lambda: defaultdict(int))
		for i in t1.keys():
			for j in t1[i].keys():
				if t1[i][j] == 1 and t2[i][j] == 1:
					t_int[i][j] = 1
		ts_int.append(t_int)
	return ts_int

# Return the union of two lists of alignment tables.
# Input:	ts1 - list of alignments as formatted by mk_align_tables()
#			ts2 - list of alignments as formatted by mk_align_tables()
# Output:	ts_int - union of tables from ts1 and ts2 
def tables_union(ts1, ts2):
	ts_int = []
	for (n, (t1, t2)) in enumerate(zip(ts1, ts2)):
		t_int = defaultdict(lambda: defaultdict(int))
		for i in t1.keys():
			for j in t1[i].keys():
				if t1[i][j] == 1:
					t_int[i][j] = 1
		for i in t2.keys():
			for j in t2[i].keys():
				if t2[i][j] == 1:
					t_int[i][j] = 1
		ts_int.append(t_int)
	return ts_int

# Top level function for alignment intersection.
# Input:	a1 - alignments as formatted by align()
#			a2 - alignments as formatted by align()
# Output:	a_int - intersection of alignments as formatted by align()
def align_intersect(a1, a2):
	ts_int = tables_intersect(mk_align_tables(a1), mk_align_tables(a2))
	return tables_to_aligns(ts_int)

# Top level function for alignment union.
# Input:	a1 - alignments as formatted by align()
#			a2 - alignments as formatted by align()
# Output:	a_int - union of alignments as formatted by align()
def align_union(a1, a2):
	ts_int = tables_union(mk_align_tables(a1), mk_align_tables(a2))
	return tables_to_aligns(ts_int)

# Prints a list of alignments in the required format. E.g.,
#     1-2 4-1 3-2
# Input: alignments - a list of row alignments
def print_output(alignments):
	for sentence in alignments:
		for (i, j) in sentence:
			sys.stdout.write("%i-%i " % (i,j))
		sys.stdout.write("\n")