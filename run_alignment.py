from aligner import *
import sys

# load the data
e_file    = "data/hansards.e"
f_file    = "data/hansards.f"

num_sents = sys.maxint
# num_sents = 10000
data      = load_input(e_file, f_file, num_sents)
rdata	  = load_input(e_file, f_file, num_sents, True)

# train the model
model = train_model(IBMM2(), data, 6)
model2 = train_model(IBMM2(), rdata, 6)

# use the model
symmetrize(align(model, data), align(model2, rdata, True))
# a1 = load_alignments("assignment1.txt")
# a2 = load_alignments("m2_all.txt")
# symmetrize(a1, a2)

#alignments = align(model, data)

# print the results
# print_output(alignments)