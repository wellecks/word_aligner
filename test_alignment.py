from aligner import *

data = [[['das', 'haus'], ['the', 'house']], [['das', 'buch'], ['the', 'book']], [['ein', 'buch'], ['a', 'book']]]

# train the model
model = train_model(IBMM1(), data)

# use the model
alignments = align(model, data)

# print the results
print_output(alignments)