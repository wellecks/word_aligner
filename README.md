##Word Aligner

####CIS526, Machine Translation, HW1

**Sean Welleck**

This project is related to the problem of aligning words from a source and target language.

The project contains three models:
- IBM Model 1
- IBM Model 2
- Bayesian Aligner

And symmetrization to combine the results of two models.

Run ```python run_alignment.py > output.txt``` to train the models and output alignments to output.txt.

-----
#####model.py
Contains the model implementations, ```IBMM1()```, ```IBMM2()```, ```BayesM()```. 

Each model extends the ```Model()``` class and must implement the ```train()``` and ```align()``` functions.
#####aligner.py
Contains top level functions for using the models:
```python
# loading data
data = aligner.load_input(e_file, f_file, num_sents)
```
```python
# training models
ibm_model1 = aligner.train_model(IBMM1(), data, num_iters)
ibm_model2 = aligner.train_model(IBMM2(), data, num_iters)
```
```python
# getting alignments using a trained model
m1_alignments = aligner.align(ibm_model1, data)
m2_alignments = aligner.align(ibm_model2, data)
```
```python
# symmetrizing output from two models
sym_alignments = aligner.symmetrize_all(m1_alignments, m2_alignments)
```
```python
# printing alignments
print_output(sym_alignments)
```
