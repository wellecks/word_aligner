### Word Aligner - Statistical Machine Translation
### Sean Welleck | 2014
#
# Contains word alignment models. Each model is a subclass of
# the abstract Model class, and must implement train() and align().

from abc import ABCMeta, abstractmethod
from collections import defaultdict
import sys

# Abstract model class. 
class Model(object):
	__metaclass__ = ABCMeta

	@abstractmethod
	def train(self, data, iters):
		pass

	@abstractmethod
	def align(self, data):
		pass

# IBM Model 1 aligner.
class IBMM1(Model):
	def __init__(self):
		# translation probabilities
		# t[i][j] = t(e_i | f_j)
		self.t       = defaultdict(lambda: defaultdict(int))
		# count[i][j] = count(e_i | f_j)
		self.count   = defaultdict(lambda: defaultdict(int)) 
		# total[f]
		self.total   = defaultdict(int)
		self.s_total = defaultdict(int)
		# english vocabulary
		self.e_vocab = set()
		# foreign vocabulary
		self.f_vocab = set()

	def train(self, data, iters):
		# initialize probabilities
		self._init_tprobs(data)

		for it in xrange(iters):
			sys.stderr.write("Iteration %i\n" % it)
			for (n, (f, e)) in enumerate(data):
				# compute normalization
				for e_j in e:
					self.s_total[e_j] = 0
					for f_i in f:
						self.s_total[e_j] += self.t[e_j][f_i]
				
				# compute counts
				for e_j in e:
					for f_i in f:
						self.count[e_j][f_i] += self.t[e_j][f_i] / self.s_total[e_j]
						self.total[f_i] += self.t[e_j][f_i] / self.s_total[e_j]

			# estimate probabilities
			for f in self.f_vocab:
				for e in self.e_vocab:
					self.t[e][f] = self.count[e][f] / self.total[f]

	def align(self, data):
		return [[(1,2)]]

	# initialize t(e|f) uniformly; i.e. to 1 / |e_vocab|
	def _init_tprobs(self, data):
		for (n, (f, e)) in enumerate(data):
			for f_i in set(f):
				self.f_vocab.add(f_i)
				for e_j in set(e):
					self.e_vocab.add(e_j)
		for f_i in self.f_vocab:
			for e_j in self.e_vocab:
				self.t[e_j][f_i] = 1.0 / len(self.e_vocab)

