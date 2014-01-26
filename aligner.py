### Word Aligner - Statistical Machine Translation
### Sean Welleck | 2014
#
# Top level module for performing word alignment.
# Contains generic functions for training and using a Model.
# Contains IO functions for loading and printing data.

from model import Model, IBMM1
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
def load_input(e_fname, f_fname, num_sents):
	bitext = [[sentence.strip().split() 
	           for sentence in pair] 
	           for pair in zip(open(f_fname), open(e_fname))[:num_sents]]
	return bitext

# Prints a list of alignments in the required format. E.g.,
#     1-2 4-1 3-2
# Input: alignments - a list of row alignments
def print_output(alignments):
	for sentence in alignments:
		for (i, j) in sentence:
			sys.stdout.write("%i-%i " % (i,j))
		sys.stdout.write("\n")