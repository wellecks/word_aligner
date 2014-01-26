from aligner import *

# load the data
e_file    = "data/hansards.e"
f_file    = "data/hansards.f"

num_sents = 10000
data      = load_input(e_file, f_file, num_sents)

# train the model
model = train_model(IBMM1(), data, 20)

# use the model
alignments = align(model, data)

# print the results
print_output(alignments)