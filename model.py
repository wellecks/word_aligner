### Word Aligner - Statistical Machine Translation
### Sean Welleck | 2014
#
# Contains word alignment models. Each model is a subclass of
# the abstract Model class, and must implement train() and align().

from abc import ABCMeta, abstractmethod
from collections import defaultdict, Counter
import sys
import pdb
import pickle

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
		# alignments
		# a[n][i] = alignment for english word i in sentence n
		self.a       = defaultdict(lambda: defaultdict(int))
		# count[i][j] = count(e_i | f_j)
		self.count   = defaultdict(int) 
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
				if n % 1000 == 0:
					sys.stderr.write("%i samples\n" % n)
				# compute normalization
				for e_j in e:
					self.s_total[e_j] = 0
					for f_i in f:
						self.s_total[e_j] += self.t[e_j][f_i]
				
				# compute counts
				for e_j in e:
					for f_i in f:
						self.count[(e_j, f_i)] += self.t[e_j][f_i] / self.s_total[e_j]
						self.total[f_i] += self.t[e_j][f_i] / self.s_total[e_j]

			# estimate probabilities
			for ((e, f), c) in self.count.iteritems():
				self.t[e][f] = c / self.total[f]

	def align(self, data):
		alignments = []
		for (n, (f, e)) in enumerate(data):
			row_alignments = []
			for (i, e_i) in enumerate(e):
				max_ind = 0
				max_amt = 0.0
				max_elt = None
				for (j, f_j) in enumerate(f):
					if self.t[e_i][f_j] > max_amt:
						max_ind = j
						max_amt = self.t[e_i][f_j]
						max_elt = f_j
				if max_elt != None:
					row_alignments.append((max_ind, i))
					self.a[n][i] = max_ind
			alignments.append(row_alignments)
		return alignments

	# initialize t(e|f) uniformly; i.e. to 1 / |e_vocab|
	def _init_tprobs(self, data):
		#self.f_vocab.add(None)
		#self.total[None] = 1
		for (f, e) in data:
			for f_i in f:
				self.f_vocab.add(f_i)
				for e_j in e:
					self.e_vocab.add(e_j)
		default_val = 1.0 / len(self.e_vocab)
		self.t = defaultdict(lambda: defaultdict(lambda: default_val))

	# save alignments so that they can be used to initialize other models
	def save_alignments(self):
		self.a.default_factory = None
		out = open('m1_align.pkl', 'wb')
		pickle.dump(self.a, out, pickle.HIGHEST_PROTOCOL)
		out.close()

# TODO: Bayesian word aligner.
class BayesM(Model):

	def __init__(self):
		self.t = defaultdict(lambda: defaultdict(int))
		self.a = defaultdict(lambda: defaultdict(int))
		self.sample = defaultdict(lambda: defaultdict(lambda: []))
		# english vocabulary
		self.e_vocab = set()
		# foreign vocabulary
		self.f_vocab = set()
		self.counts = defaultdict(lambda: defaultdict(int))
		self.totals = defaultdict(int)

	def train(self, data, iters):
		# Gibbs Sampling parameters
		burn_in   = 5
		num_samps = 3
		padding   = 1

		self.gibbs_sample(data, burn_in, num_samps, padding)
		return []

	def gibbs_sample(self, data, B, M, L):
		samples = []

		# initialize priors
		self._init_tpriors(data)
		self._init_apriors(data)

		num_iters = B + (M * L) + 1
		for it in xrange(num_iters):
			sys.stderr.write("Iteration %i\n" % it)
			keep_sample = (it > B) and (it - B) % L == 0
			for (n, (f, e)) in enumerate(data):
				for (j, e_j) in enumerate(e):
					# probability distribution for aj
					p_aj = []

					self.counts[e_j][self.a[n][j]] -= 1
					self.totals[e_j] -= 1
					
					# fill the distribution with the potential choices
					for (i, f_i) in enumerate(f):
						p_aj.append(self._gibbs_prob(j, e_j, i, f_i))

					# sample a value from the distribution	
					# index of french word in f
					result_index = self._sample_value(p_aj)
					self.a[n][j] = result_index
					# count the sample as an alignment observation
					self.counts[e_j][f[result_index]] += 1
					self.totals[e_j] += 1

					if keep_sample:
						self.sample[n][j].append(result_index)

	def _sample_value(self, distribution):
		distribution = map(lambda x: round(x*10000), distribution)
		s = sum(distribution)
		i = -1
		while s > 0:
			i += 1
			s -= distribution[i]
		return i


	def _gibbs_prob(self, j, e_j, i, f_i):
		p_aji = (self.counts[e_j][f_i] + self.t[e_j][f_i]) / (self.totals[e_j] + len(self.f_vocab)*self.t[e_j][f_i])
		return p_aji

	def align(self, data):
		alignments = []
		for (n, (f, e)) in enumerate(data):
			row_alignments = []
			for (i, e_i) in enumerate(e):
				counter = Counter(self.sample[n][i])
				max_ind = counter.most_common(1)[0][0]
				max_elt = f[max_ind]
				if max_elt != None:
					row_alignments.append((max_ind, i))
					self.a[n][i] = max_ind
			alignments.append(row_alignments)
		return alignments

	# initialize priors for t(e|f)
	def _init_tpriors(self, data):
		for (f, e) in data:
			for f_i in f:
				self.f_vocab.add(f_i)
				for e_j in e:
					self.e_vocab.add(e_j)
		# sparse Dirichlet prior
		default_val = 0.0001
		self.t = defaultdict(lambda: defaultdict(lambda: default_val))

	# initalize priors for a[sentence][word]
	# use output of IBMM1 model
	def _init_apriors(self, data):
		f = open('m1_align.pkl')
		a = pickle.load(f)
		f.close()
		pdb.set_trace()
		a = defaultdict(lambda: defaultdict(lambda: -1), a)
		self.a = a
		for (n, (f, e)) in enumerate(data):
			for (j, e_j) in enumerate(e):
				mapping_index = self.a[n][j]
				if mapping_index != -1:
					self.counts[e_j][f[mapping_index]] += 1
				self.totals[e_j] += 1

