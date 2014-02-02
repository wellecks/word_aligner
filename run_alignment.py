from aligner import *
import sys

# load the data
e_file    = "data/hansards.e"
f_file    = "data/hansards.f"

#num_sents = sys.maxint
num_sents = 1000
data      = load_input(e_file, f_file, num_sents)
rdata	  = load_input(e_file, f_file, num_sents, True)

# train the model
model = train_model(IBMM1(), data, 10)
model2 = train_model(IBMM1(), rdata, 10)

# use the model
#alignments  = align_intersect(align(model, data), align(model2, rdata))
alignments = align_union(align(model, data), align(model2, rdata))

# model.save_alignments()

# print the results
print_output(alignments)