from aligner import *
import sys

# load the data
e_file    = "data/hansards.e"
f_file    = "data/hansards.f"

num_sents = sys.maxint
data      = load_input(e_file, f_file, num_sents)

# train the model
model = train_model(BayesM(), data, 10)

# use the model
alignments = align(model, data)

# model.save_alignments()

# print the results
print_output(alignments)