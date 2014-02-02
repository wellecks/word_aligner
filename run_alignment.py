from aligner import *
import sys

# load the data
e_file    = "data/hansards.e"
f_file    = "data/hansards.f"

num_sents = sys.maxint
#num_sents = 1000
data      = load_input(e_file, f_file, num_sents)
rdata	  = load_input(e_file, f_file, num_sents, True)

# train the model
model = train_model(IBMM2(), data, 5)
model2 = train_model(IBMM2(), rdata, 5)

# use the model
#alignments  = align_intersect(align(model, data), align(model2, rdata))
#alignments = align_union(align(model, data), align(model2, rdata))
alignments = symmetrize(align(model, data), align(model2, rdata, True))
#alignments = align(model, data)
# model.save_alignments()

# print the results
print_output(alignments)