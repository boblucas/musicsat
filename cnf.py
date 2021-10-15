from itertools import *
from collections import *
from math import *

from sat_util import *
from music import *

class Solution:
	def __init__(self, time, voices, progression, transposition, model):
		# store data in easy to access format
		self.time = time
		self.model = dict((abs(v), v > 0) for v in model)
		self.progression = progression
		self.voices = [[(
			dia_from_chroma([self.model[x] for x in note.pitch].index(True)),
			-chroma_is_flat([self.model[x] for x in note.pitch].index(True)),
			note.dissonance_name(self.model)) for note in voice] for voice in voices]
		self.intervals = [[([self.model[x] for x in note.intervals] + [True]).index(True) for note in voice] for voice in voices]

		self.transposition = [[self.model[t] for t in transpose].index(True) for transpose in transposition]
		self.compute_statistics()

	def compute_statistics(self):
		r,d = self.voices[0], self.voices[2]
		self.static = sum([abs(x[0]-y[0]) == 0 for x,y in list(zip(r, r[1:] + r[:1])) + list(zip(d, d[1:] + d[:1]))])
		self.steps = sum([abs(x[0]-y[0]) in [1,6]  for x,y in list(zip(r, r[1:] + r[:1])) + list(zip(d, d[1:] + d[:1]))])
		self.variety = prod(Counter([x for x,y,_ in self.voices[0]]).values())
		self.total_notes = 64-self.static
		self.score = log(1+(self.steps)**2*self.total_notes**2)

	def __str__(self):
		s = ''
		for v,t,o,i in zip(self.voices, [0] + self.transposition, [0] + list(self.time), self.intervals):
			s = f"{s}{['   ', '2nd', '3th', '4th', '5th', '6th', '7th'][t]} -{o:2}m {''.join(['cdefgab'[n] + ({-1:'♭',0:' ',1:'♯'}[a]) for n,a,d in v])}\n"
			s = f"{s}         {''.join(d for n,a,d in v)}\n"
			#s = f"{s}interval {' '.join(str(a) for a in i)}\n"

		s = s + f'stepcount = {self.steps}\n'
		s = s + f'variety   = {self.variety:.3f}\n'
		s = s + f'note_count= {self.total_notes}\n'
		s = s + f'score     = {self.score:.3f}'
		return s


class Note:
	def __init__(self, problem, strong_beat, bass, name):
		self.name = name
		self.strong_beat = strong_beat
		self.bass = bass
		self.pitch = [problem.make_var(f'{name}-p{c}') for c in range(12)]
		self.intervals = []
		self.suspension = problem.make_var()
		self.passing = problem.make_var()
		problem.add_clause(self.pitch)

	def pitches_for_dia(self, p): return [self.pitch[x] for x in chromas_from_dia(p)]
	def dissonance(self): return self.suspension if self.strong_beat else self.passing

	def dissonance_name(self, solution):
		return {self.suspension: 'S-', self.passing: 'P ', None: '  '}[self.dissonance() if not self.dissonance() is None and solution[self.dissonance()] else None]

# we really only need -1, 0, 1, <anything else>
def store_intervals(problem, a, b):
	intervals = [problem.make_var(f'{a.name} {b.name} +{i}') for i in range(8)]
	problem.atmost(1, intervals)
	for i,j in product(range(12), range(12)):
		d = (dia_from_chroma(j) - dia_from_chroma(i)) % 7
		if d == 0 and i != j: d = 7
		problem.add_clause([-a.pitch[i], -b.pitch[j], intervals[d]])

	return intervals

def store_intervals_stepwise(problem, a, b):	
	intervals = [problem.make_var(), problem.make_var(), problem.make_var(), problem.make_var(), problem.make_var()]

	for i in range(12):
		for j in [-2, -1, 0, 1, 2]:
			problem.add_clause([-a.pitch[i], -b.pitch[(i+j)%12], intervals[j]])
			problem.add_clause([a.pitch[i], -b.pitch[(i+j)%12], -intervals[j]])
			problem.add_clause([-a.pitch[i], b.pitch[(i+j)%12], -intervals[j]])

	diatonic = [intervals[0], problem.make_var(), problem.make_var()]
	problem.or_connect(intervals[1], intervals[2], diatonic[1])
	problem.or_connect(intervals[-1], intervals[-2], diatonic[-1])
	return diatonic


# limits the allowed chromatic interval between two diatonic interval, here is the no-bueno table (symmetric on the other side)
# Gb: B E A D   Db: B E A   Ab: B E  Eb: B  Bb:  F :
def melodic_relationship(problem, a, b):
	for i,j in product(range(12), range(12)):
		#1 < abs(i-j)%12 < 11 and 
		if is_aug_or_dim(i, j):
			problem.add_clause([-a.pitch[i], -b.pitch[j]])
	
	# and forbid tritones too
	for i in range(12):
		problem.add_clause([-a.pitch[i], -b.pitch[(i+6)%12]])

def forbid_false_relations(problem, a, b):
	for c,d in product(range(12), range(12)):
		if 1 < abs(c-d)%12 < 11 and is_aug_or_dim(c, d):
			problem.add_clause([-a.pitch[c], -b.pitch[d]])

# The only tritones are
# Gb C -> Db F
# G Db -> Ab C
# Ab D -> Eb Gb
# A Eb -> Bb Db
# Bb E -> F Ab
# the one on the tonic is special, though we will omit the modulation to Gb
# B  F -> E C, Eb C, Gb A
# also note that any tritone can be repeated
def resolve_tritone(problem, a1, a2, b1, b2):
	for v,w,x,y in [(a1,a2,b1,b2), (a2,a1,b2,b1)]:
		for p in range(5):
			problem.add_clause([-v.pitch[p], -w.pitch[p + 6], x.pitch[p],   x.pitch[p+1 if p % 2 == 0 else p-2]])
			problem.add_clause([-v.pitch[p], -w.pitch[p + 6], y.pitch[p+6], y.pitch[(p+6)-2 if p % 2 == 0 else (p+6)+1]])
			problem.add_clause([-v.pitch[p], -w.pitch[p + 6], -x.pitch[p],   -y.pitch[(p+6)-2 if p % 2 == 0 else (p+6)+1]])
			problem.add_clause([-v.pitch[p], -w.pitch[p + 6], -y.pitch[p+6], -x.pitch[p+1 if p % 2 == 0 else p-2]])

		problem.add_clause([-v.pitch[5], -w.pitch[11], y.pitch[11], y.pitch[0]])
		problem.add_clause([-v.pitch[5], -w.pitch[11], x.pitch[5], x.pitch[3], x.pitch[4]])
		problem.add_clause([-v.pitch[5], -w.pitch[11], -x.pitch[5], -y.pitch[0]])
		problem.add_clause([-v.pitch[5], -w.pitch[11], -y.pitch[11], -x.pitch[3]])
		problem.add_clause([-v.pitch[5], -w.pitch[11], -y.pitch[11], -x.pitch[4]])

N = 4

def solve_for_time(time):
	file = open('solutions_' +  '_'.join(map(str, time)), 'w')
	problem = Problem()

	voices = [[Note(problem, n % 2 == v % 2, v == 3, f'{v}-{n}' ) for n in range(N*N*2)] for v in range(4)]
	inverses, reverses, verticals = [1,0,1], [1,0,1], [0,1,1]

	transposition = [[problem.make_var() for p in range(7)] for voice in voices[1:]]
	for t in transposition: problem.exactly(1, t)

	# transposition is diatonic, that means that if one 'pair of pitches' in one measure is selected
	# than you can choose from a mapped pair of pitches in another measure
	# eg: ((a | b) = (c | d)) | !i
	# eq: (a | b | !c | !i) & (a | b | !d | !i) & (c | d | !a | !i) & (c | d | !b | !i)
	for i,p,q in product(range(N*N*2), range(7), range(7)):
		for v, inv, t in zip(voices[1:], inverses, transposition):
			ab = voices[0][i].pitches_for_dia(p)
			cd = v[i].pitches_for_dia(6-q if inv else q)
			for x in cd: problem.add_clause(ab + [-x, -t[(q-p)%7]])
			for x in ab: problem.add_clause(cd + [-x, -t[(q-p)%7]])

	# now apply time transformations by shuffling actual arrays of variables
	voices = [voices[0]] + [rotate((columns(voice, N, 2) if vertical else voice)[::reverse*-2+1], t) for voice, reverse, vertical, t in zip(voices[1:], reverses, verticals, time)]

	# and then we can simply create a matrix that we can query without caring about the above transforms
	# all the following code doesn't care about the canonic form
	progression = list(zip(*voices))
	contexts = [list(zip(rotate(v, -1), v)) for v in voices]

	for c in contexts:
		for x,y in c:
			x.intervals = store_intervals_stepwise(problem, x, y)

	for c in contexts:
		for x,y in c:
			melodic_relationship(problem, x, y)

		for x,y in c[::2]:
			problem.add_clause([-y.suspension, x.intervals[0]])
			problem.add_clause([-y.suspension, y.intervals[-1]])
			problem.add_clause([y.suspension, -x.intervals[0], -y.intervals[-1]])

		for x,y in c[1::2]:
			problem.add_clause([-y.passing, x.intervals[1], x.intervals[-1]])
			problem.add_clause([-y.passing, y.intervals[1], y.intervals[-1]])
			problem.add_clause([y.passing, -x.intervals[1], -y.intervals[-1]])
			problem.add_clause([y.passing, -x.intervals[-1], -y.intervals[1]])
			problem.add_clause([y.passing, -x.intervals[1], -y.intervals[1]])
			problem.add_clause([y.passing, -x.intervals[-1], -y.intervals[-1]])

	# forbid combining resolving supensions and passing tones within a single measure
	# you may combine multiple suspensions, or have multiple passing tones however
	for strong,weak in zip(progression[::2], progression[1::2]):
		for s,w in product(strong, weak):
			problem.add_clause([-s.dissonance(), -w.dissonance()])
		
		# no double suspensions
		problem.atmost(1, [s.dissonance() for s in strong])
		# no triple passing/neighbours
		problem.atmost(2, [s.dissonance() for s in weak])


	# is weakly true but strongly false
	harmonic_vectors = [[problem.make_var() for _ in range(12)] for _ in range(len(progression)//2)]
	for i,strong,weak in zip(range(len(progression)//2), progression[::2], progression[1::2]):
		for a,b in combinations(strong + weak, 2):
			forbid_false_relations(problem, a, b)

		for p in range(12):
			for note in strong[:-1] + weak[:-1]:
				problem.add_clause([harmonic_vectors[i][p], -note.pitch[p], note.dissonance()])

			# this is not technically correct as some of these notes may be dissonants
			problem.add_clause([-harmonic_vectors[i][p]] + [note.pitch[p] for note in strong[:-1] + weak[:-1]])
	
	for i,strong,weak in zip(range(len(progression)//2), progression[::2], progression[1::2]):
		h = harmonic_vectors[i]
		for note in strong[-1:] + weak[-1:]:
			for p in range(12):
				# 1: no 1,2,5,11 or semitone distances, no 2nd inversion or M7 chords
				problem.add_clause([note.dissonance(), -note.pitch[p], -h[p-11]])
				problem.add_clause([note.dissonance(), -note.pitch[p], -h[p-10]])
				problem.add_clause([note.dissonance(), -note.pitch[p], -h[p- 7]])
				problem.add_clause([note.dissonance(), -note.pitch[p], -h[p- 1]])
				
				# 2: no semitones, no '4 8', all augemented chords are forbidden, symmetrical so this suffices
				# 3: no '4 6', no 3th inversions (the tritone defines the tonic) 
				# 4: no '7 9', no minor 7 chords
				# 5: no '8 10', that's just 3 2nds
				for i,j in [(3,4),(6,7),(7,8),(8,9),(9,10),(4,6),(7,9),(8,10)]:
					problem.add_clause([note.dissonance(), -note.pitch[p], -h[(p+i)%12], -h[(p+j)%12]])

				# 5: no 10 wo. 4, no bare minor 7ths
				problem.add_clause([note.dissonance(), -note.pitch[p], -h[(p+10)%12], h[(p+4)%12]])

	# none of the movements between 2 pairs of notes may be parallel perfects
	for notes_a, notes_b in zip(progression, progression[1:] + progression[:1]): # 16
		for a,b in combinations(list(zip(notes_a, notes_b)), 2): # 6
			for i,j in product(range(12), range(12)):
				if i != j:
					problem.atleast(1, [-a[0].pitch[(i+7)%12], -b[0].pitch[i], -a[1].pitch[(j+7)%12], -b[1].pitch[j]])
					problem.atleast(1, [-a[0].pitch[(i-2)%12], -b[0].pitch[i], -a[1].pitch[(j-2)%12], -b[1].pitch[j]])
					problem.atleast(1, [-a[0].pitch[i], -b[0].pitch[i], -a[1].pitch[j], -b[1].pitch[j]])

	# and harmonically, although here we allow them to resolve without preparation
	# as a nice V7 -> I is too good to pass up really
	for a, b in zip(progression, progression[1:] + progression[:1]): # 16
		for x,y in combinations(zip(a, b), 2):
			resolve_tritone(problem, x[0], y[0],  x[1], y[1])

	# now let's do some more subjective minimal quality things
	# 1/3 stepwise motion is not a lot but works quite well in filtering a lot of the non-solutions
	problem.atleast(24, [x.intervals[i] for x in voices[0] + voices[2] for i in (1, -1)])
	# having a few suspensions really adds something too, this one filters super-hard though
	problem.atleast(1, [x.suspension for x in voices[0] + voices[2] + voices[1] + voices[3] if x.strong_beat])

	for model in problem.solutions([x for note in voices[0] for x in note.pitch]):
		solution = Solution(time, voices, progression, transposition, model)
		file.write(str(solution) + '\n')
		file.flush()

	return True

from multiprocessing import *
if __name__ == '__main__':
    #solve_for_time((0,0,0))
    with Pool(processes=32) as pool:
    	result = list(pool.imap_unordered(solve_for_time, product(list(range(0, N*N*2, 2)), list(range(0, N*N*2, 2)), list(range(0, N*N*2, 2)))))
