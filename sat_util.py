from pysat.solvers import *
from pysat.card import *
from pysat.formula import IDPool
from itertools import *
from collections import *
from math import *

# add CNF clauses & variables trough somewhat higher level means
class Problem:
	def __init__(self):
		self.pool = IDPool()
		self.topid = 0
		self.clauses = []

	def make_var(self, name = None):
		if name is None:
			self.topid = self.pool.id()
		else:
			self.topid = self.pool.id(name)
		return self.topid
		
	def add_clause(self, C, name = ''):
		if not C: 
			return
		
		for c in (C if type(C[0]) == list else [C]):
			self.clauses.append((c, name))

	def atleast(self, n, V): self.add_clause(CardEnc.atleast(lits=V, bound=n, vpool=self.pool, encoding=EncType.seqcounter).clauses)
	def atmost (self, n, V): self.add_clause(CardEnc.atmost(lits=V, bound=n, vpool=self.pool, encoding=EncType.seqcounter).clauses)
	def exactly(self, n, V): self.add_clause(CardEnc.equals(lits=V, bound=n, vpool=self.pool, encoding=EncType.seqcounter).clauses)
	def all(self, V): return atleast(len(V), V)

	def equal(self, a, b): self.add_clause([[a, -b], [-a, b]])
	def not_equal(self, a, b): self.add_clause([[a, b], [-a, -b]])
	def all_equal(self, V, W): [self.equal(a,b) for a,b in zip(V, W)]

	def atmost_store(self, V, n):
		v = self.make_var()
		clauses = CardEnc.atmost(lits=V, bound=n, vpool=self.pool, encoding=EncType.seqcounter).clauses
		self.add_clause([[-v] + c for c in clauses])
		self.add_clause([[v] + [-w for w in c] for c in clauses])
		return v

	def atleast_store(self, V, n):
		v = self.make_var()
		clauses = CardEnc.atleast(lits=V, bound=n, vpool=self.pool, encoding=EncType.seqcounter).clauses
		self.add_clause([[-v] + c for c in clauses])
		self.add_clause([[v] + [-w for w in c] for c in clauses])
		return v

	def exactly_store(self, V, n):
		v = self.make_var()
		clauses = CardEnc.equals(lits=V, bound=n, vpool=self.pool, encoding=EncType.seqcounter).clauses
		self.add_clause([[-v] + c for c in clauses])
		self.add_clause([[v] + [-w for w in c] for c in clauses])
		return v

	# c = a & b
	def and_connect(self, a, b, c): self.add_clause([[-a, -b,  c], [-a,  b, -c], [ a, -b, -c], [ a,  b, -c]])
	# c = a | b
	def or_connect(self, a, b, c): self.add_clause([[a, b, -c], [-a, c], [-b, c]])

	# return = all(V)
	def and_store(self, V): return self.atleast_store(V, 0)[0]

	# instead of negating the whole solution there is problably some small defining subset of variables that define the essence of the solution
	def solutions(self, exclusion_vars = []):
		solver = Glucose3()
		for clause, name in self.clauses:
			solver.add_clause(clause)
		while has_solution := solver.solve():
			model = solver.get_model()
			yield model
			solver.add_clause([-x if x in model else x for x in (exclusion_vars if exclusion_vars else model)])
