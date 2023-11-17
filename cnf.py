from music import *
from sat_util import *
from itertools import *
from collections import *
from dataclasses import dataclass, field
from typing import List
from music21 import *

MEASURE_TUPLE = 2
BEAT_TUPLE = 1
#COF_LO, COF_HI = -1, 5+1 # diatonic scale
#COF_LO, COF_HI = -6, 5+1 # chromatic scale (all flat)
COF_LO, COF_HI = -6, 9+1 # Db - D#, usually enough
# COF_LO, COF_HI = -7, 14 # Cb - Fx, pretty much all practically used notes

COFS = list(range(COF_LO, COF_HI))

########
## Equivalence under transformation
########
# define equivalence between voices vector of bitsets of pitches under dynamic time & pitch transformations
# this gives you the tools to set up some canonic form with variance in time and pitch transforms
# and all further code can just evaluate this 2D pitch-bitset array as if it's a static thing.
##################################

@dataclass
class PitchTransform:
	# list of legal transposition intervals in CoF notation
	valid_transpositions:List[int] = field(default_factory=list)
	# 7 = diatonic, 12 = enharmonic, 0 = strict equivalance
	mod:int = 7
	# 1 is regular, -1 inversed
	scale:List[int] = field(default_factory=list)

@dataclass
class TimeTransform:
	rotations:List[int] = field(default_factory=list)
	scales:List[int] = field(default_factory=list)
	reorderings:List[List[int]] = field(default_factory=list)

@dataclass
class Chord:
	# result from create_pitch, chroma bitset
	root:dict = field(default_factory=dict)
	chroma:dict = field(default_factory=dict)

	# boolean variables that describe all valid chords
	major:int = field(default_factory=int)
	seven:int = field(default_factory=int)
	dim:int = field(default_factory=int)

def create_pitch(problem, lo = None, hi = None):
	if lo is None: lo, hi = COF_LO, COF_HI
	pitch = {c:problem.make_var() for c in range(lo, hi)}
	pitch['rest'] = problem.make_var()
	problem.exactly(1, list(pitch.values()))
	return pitch

def create_chord(problem):
	lo, hi = COF_LO, COF_HI
	pitch = {c:problem.make_var() for c in range(lo, hi)}
	pitch['rest'] = problem.make_var()
	problem.exactly(1, list(pitch.values()))
	return pitch

def equivalent_under_pitch_transform(problem, a, b, transposition, rep, distance, short_circuit = []):
	transposition = {rep(k):v for k,v in transposition.items()}
	classes = {x:set(y) for x,y in groupby(sorted(range(-7, 12), key=rep), key=rep)}
	for g1, g2 in product(classes.items(), classes.items()):
		if (d := rep(distance(g1[0], g2[0]))) in transposition:
			g1 = [a[x] for x in g1[1] if x in a]
			g2 = [b[x] for x in g2[1] if x in b]
			for x in g2: problem.add_clause(g1 + [-x, -transposition[d]] + short_circuit)
			for x in g1: problem.add_clause(g2 + [-x, -transposition[d]] + short_circuit)

def linear_pitch_equivalence(problem, a, b, transposition, pitch_transform, offset, short_circuit = []):
	repr_pitch = (lambda x: x%pitch_transform.mod if pitch_transform.mod else x)
	for scale in pitch_transform.scale:
		t = {k[0]+offset:v for k,v in transposition.items() if k[1] == scale}
		equivalent_under_pitch_transform(problem, a, b, t, repr_pitch, (lambda x,y: y*scale - x), short_circuit)

	# regardless of the transposition a rest remains a rest
	problem.add_clause([-a['rest'], b['rest']] + short_circuit)
	problem.add_clause([a['rest'], -b['rest']] + short_circuit)

def apply_time_transform_vector(v, rotate, scale, reorder):
	v = [v[j] for j in reorder]
	v = (v[rotate:] + v[:rotate])[::max(-1, min(scale, 1))]
	v = list(chain(*[[x]*abs(scale) for x in v]))
	return v

# the time transforms talk about rotation but in practice this means having an infinite canons
# now rotation doesn't map onto infinite modulation canons as the repeats are not the same
# this is what the modulation parameter is for, it allows you to specify how the rotated section
# of the theme is transposed in addition to the the transformation after pitch_transform
def transformational_equivalence(problem, a, pitch_transform, time_transform, modulation = 0, add_extra = 0):
	# intermediate variabels for the equivalent voice
	scale = max(map(abs, time_transform.scales))
	b = [create_pitch(problem) for _ in range(len(a)*scale + add_extra)]
	
	if not time_transform.reorderings:
		time_transform = TimeTransform(time_transform.rotations, time_transform.scales, [list(range(len(a)))])

	# variables for controlling transform
	transposition = {(c,s):problem.make_var() for c,s in product(pitch_transform.valid_transpositions, pitch_transform.scale)}
	time = {(r,s,tuple(o)): problem.make_var() for r,s,o in product(time_transform.rotations, time_transform.scales, time_transform.reorderings)}

	#problem.add_clause(list(transposition.values()))
	#problem.add_clause(list(time.values()))
	
	problem.exactly(1, list(transposition.values()))
	problem.exactly(1, list(time.values()))

	# specify equivalence
	for t,t_var in time.items():
		at = apply_time_transform_vector(a, *t)
		if add_extra:
			at += (at*8)[:add_extra]
		
		rs = t[0]*t[1]
		for i,(p,q) in enumerate(zip(at, b)):
			linear_pitch_equivalence(problem, p, q, transposition, pitch_transform, modulation*(i >= (len(a)*scale-rs)), [-t_var])

	return (transposition, time, b)

###########
# Reasonable melodic lines
###########

# extract specific pitches from voice
def context(x, n): return list(zip(*[x[i:] for i in range(n)]))
def groupn(x, n): return [x[i:i+n] for i in range(0, len(x), n)]
def beats(voice): return groupn(voice, BEAT_TUPLE)
def harmonies(voices): return list(zip(*voices))
def beat_notes(voice): return voice[::BEAT_TUPLE]

# note to note rules
# rest_allowed = 0, no rests
# rest_allowed = 1, any rests are ok
# rest_allowed = 2, rests are ok but only 0 or 2 rests
# rest_allowed = 3, `b` can only be a rest if `a` is a rest
def enforce_interval(problem, a, b, motion, except_when = [], rest_allowed = 1):
	problem.add_clause([[*except_when, b['rest'], -a[i], *[b[i+j] for j in motion if i+j in COFS]] for i in COFS])

def stepwise(problem, a, b, except_when = [], rest_allowed = 1):
	enforce_interval(problem, a, b, [0, -2,2,-5,5], except_when, rest_allowed)

def no_false_relations(problem, a, b, tritones = False, half_tones = True, whole_tones = True, except_when = []):
	intervals = [-4,-3,-1,0,1,3,4] + [2,-2]*whole_tones + [-5,5]*half_tones + [-6,6]*tritones
	#intervals = [i+j for i in intervals for j in range(-24,25,12)]
	enforce_interval(problem, a, b, intervals, except_when, 1)

def create_dux(problem, duration):
	return [create_pitch(problem) for _ in range(duration)]

def ornaments_stepwise(problem, a): [stepwise(problem, f, t, rest_allowed = 2) for beat in beats(a) for f,t in zip(beat, beat[1:])]
def downbeats_diatonic(problem, a): [no_false_relations(problem, p, q) for p,q in zip(beat_notes(a), beat_notes(a)[1:])]
def ornaments_diatonic(problem, a): [no_false_relations(problem, p, q) for p,q in zip(a, a[1:] + a[:1])]
def from_ornament_to_downbeat_stepwise(problem, a):
	for x,y in context(beats(a), 2):
		# x[0] == x[-1] || stepwise(x[-1], y[0])
		problem.add_clause([[x[-1]['rest'], y[0]['rest'], x[0][i], -x[-1][i], *[y[0][i+j] for j in [0]*(BEAT_TUPLE > 2) + [-2,2,-5,5] if i+j in COFS]] for i in COFS])

def reasonable_progression(problem, chords, except_when=[]):
	for a,b in context(chords, 2):
		# only active when minor -> minor
		enforce_interval(problem, a.root, b.root, [-2,-1,1,2,3,4], [a.major, b.major], 1)
		# only active when major -> minor
		enforce_interval(problem, a.root, b.root, [-2,-1,1,2,3,4], [-a.major, b.major], 1)
		# only active when minor -> major
		enforce_interval(problem, a.root, b.root, [-4,-3,-2,-1,1], [a.major, -b.major], 1)
		# only active when major -> major
		enforce_interval(problem, a.root, b.root, [-4,-3,-2,-1,1], [-a.major, -b.major], 1)

########
# Contrapuntal & harmonic rules
#########

# marks downbeats that follow the form of a suspense and upbeats that follow the form of a NT or PT
def are_beats_valid_dissonances(problem, a):
	dissonance = [problem.make_var()]
	problem.add_clause([-dissonance[0]])
	for x,y,z in context(beats(a), 3):
		dissonance.append(problem.make_var())
		# sus & pt/nt
		if (len(dissonance)-1) % MEASURE_TUPLE == 0:
			enforce_interval(problem, x[0], y[0], [0], [-dissonance[-1]], 0)
			if len(x) > 1: enforce_interval(problem, x[-1], y[0], [0], [-dissonance[-1]], 0)
			enforce_interval(problem, y[0], z[0], [-2, 5], [-dissonance[-1]], 0)
		else:
			enforce_interval(problem, x[0], y[0], [2,-2,-5,5], [-dissonance[-1]], 0)
			enforce_interval(problem, y[0], z[0], [2,-2,-5,5], [-dissonance[-1]], 0)

	return dissonance

# given some harmony and a bass note returns pitch vector of root of chord
def get_root(problem, harmony, bass, except_when):
	root = create_pitch(problem, COF_LO-4, COF_HI+3)
	problem.add_clause([root['rest']] + [-x for x in except_when])

	for c in COFS:
		problem.add_clause([-bass[c], root[c], root[c+3], root[c-4]] + except_when)
		if c+1 in harmony: problem.add_clause([-bass[c], -harmony[c+1], root[c]] + except_when)
		if c+3 in harmony: problem.add_clause([-bass[c], -harmony[c+3], root[c+3]] + except_when)
		if c-4 in harmony: problem.add_clause([-bass[c], -harmony[c-4], root[c-4]] + except_when)

	return root

# notes should be a tuple of (bitset, is_bass, short_circuit)
def harmonic_vector(problem, _notes):
	notes = list(chain(*_notes))

	h = {c:problem.make_var() for c in range(-8,12)}
	[no_false_relations(problem, a[0], b[0], True, False, True) for a,b in combinations(_notes[0], 2)]
	[no_false_relations(problem, a[0], b[0], True, True, True) for a,b in combinations(notes, 2)]

	# all non-dissonant notes make up part of the harmony
	for p in COFS:
		treble = [note for note in notes if not note[1]]
		# note_in_harmony != note_is_dissonant
		problem.add_clause([[h[p], -note[0][p], note[2]] for note in treble])
		problem.add_clause([[-h[p], -note[0][p], -note[2]] for note in treble])
		problem.add_clause([-h[p]] + [note[0][p] for note in treble])
	
	# the consonant bass note has special relationship to all harmony notes
	for note in [note for note in notes if note[1]]:
		for p in COFS:
			# 1.1: -1 no 2nd inversions
			# 1.2:  2 no 3th inversions
			# 1.3:  5 no M7
			# 1.4:  7 no m2
			# optionally: no tritones (eg: no V7)
			for i in [-1,  2,5,7,10, -2,-5,-7,-10, 6,-6]:
				if p+i in h:
					problem.add_clause([note[2], -note[0][p], -h[p+i]])

			# -3  4 no minor and major 3th combined
			#  6  1 no A4 and P5
			#  1 -4 no m6 and P5
			# -4  3 no m6 and M6
			#  3 -2 no M6 and m7
			#  4  6 no M3 and A4
			#  1  3 no M6 and P5, eg no m3 m7 chords
			# -4 -2 no m6 and m7
			invalids = [(-3,4),(6,1),(1,-4),(-4,3),(3,-2),(4,6),(1,3),(-4,-2)]
			invalids = [(x[0]+a*12, x[1]+b*12) for x in invalids for a in [-1,0,1] for b in [-1,0,1]]
			for i,j in invalids:
				if p+i in h and p+j in h:
					problem.add_clause([note[2], -note[0][p], -h[p+i], -h[p+j]])

			# 5: no 10 wo. 4, no bare minor 7ths
			if p-2 in h and p+4 in h:
				problem.add_clause([note[2], -note[0][p], -h[p-2], h[p+4]])

	# at least one bass note must be non-dissonant so we can combine all roots to get one true root
	roots = [get_root(problem, h, note, [is_dissonant]) for note,is_bass,is_dissonant in notes if is_bass]
	root = create_pitch(problem)
	#problem.add_clause([-root['rest']])
	for p in set(root.keys()) - {'rest'}:
		problem.add_clause([[root[p], -note[p], note['rest']] for note in roots])
		problem.add_clause([-root[p]] + [note[p] for note in roots])
		problem.add_clause([-root[p], h[p]])

	major = problem.make_var()
	seven = problem.make_var()
	dim = problem.make_var()

	for p in set(root.keys()) - {'rest'}:
		if p+4 in h: problem.add_clause([-root[p], -h[p+4], major])
		if p-3 in h: problem.add_clause([-root[p], -h[p-3], -major])

		if p+5 in h: problem.add_clause([-root[p], -h[p+5], seven])
		if p-2 in h: problem.add_clause([-root[p], -h[p-2], seven])
		if p+5 in h and p-2 in h: problem.add_clause([-root[p], h[p+5], h[p-2], -seven])

		if p+6 in h: problem.add_clause([-root[p], -h[p+6], dim])
		if p-6 in h: problem.add_clause([-root[p], -h[p-6], dim])
		if p+6 in h and p-6 in h: problem.add_clause([-root[p], h[p+6], h[p-6], -dim])

	return Chord(root = root, chroma = h, major = major, seven = seven, dim = dim)

# a is the higher voice, forbids parallel fifths and octaves
def forbid_parallel_motion(problem, a, b, motion = [0,1]):
	for x,y in zip(harmonies([a, b]), harmonies([a, b])[1:]):
		for m in motion:
			for p,q in product(range(COF_LO+m, COF_HI), range(COF_LO+m, COF_HI)):
				if p != q:
					problem.add_clause([-x[0][p], -x[1][p-m], -y[0][q], -y[1][q-m]])

# consec. dissonances are allowed in some cases most notably in contrary stepwise motion
# or in telescoped resolutions, but those cases have many conditions and exceptions
# and since most of our contrapuntal infrastructure is focussed on the half note (eg 1st, 2nd 4th species)
# blanket fixing 3th species is an easy out
def forbid_consencutive_dissonances(problem, a, b):
	is_dissonant = []

	for x,y in zip(a,b):
		is_dissonant.append(problem.make_var())
		enforce_interval(problem, x, y, [-6,-4,-3,-1,0,1,3,4,6], [is_dissonant[-1]])

	for x,y in zip(is_dissonant, is_dissonant[1:]):
		problem.add_clause([-x, -y])

	return is_dissonant

def resolve_single_tritone(problem, a,b, c,d):
	for p in range(COF_LO, COF_HI - 6):
		problem.add_clause([-a[p], -b[p+6], d[p+1]])
		problem.add_clause([-a[p], -b[p+6], c[p+4]] + ([c[p-3]] if p-3 in COFS else []))

def resolve_tritones(problem, a,b, c,d):
	resolve_single_tritone(problem, a,b, c,d)
	resolve_single_tritone(problem, b,a, d,c)

def follow_rules_of_counterpoint(problem, voices):
	# 1st 2nd 4th species
	is_dissonance  = [are_beats_valid_dissonances(problem, voice) for voice in voices]
	annotated_beats = [[(note, i == len(voices)-1, d) for note,d in zip(beat_notes(voice), is_dissonance[i]) ] for i,voice in enumerate(voices)]
	harmonies = [harmonic_vector(problem, harmony) for harmony in groupn(list(zip(*annotated_beats)), MEASURE_TUPLE) ]
	
	#[downbeats_diatonic(problem, voice) for voice in voices]
	[resolve_tritones(problem, *x, *y) for a,b in combinations(voices, 2) for x,y in context(list(zip(beat_notes(a), beat_notes(b))), 2)]
	[forbid_parallel_motion(problem, a, b) for a,b in combinations(voices, 2)]

	# bass or treble is dissonant, but not both
	for vert in zip(*is_dissonance): problem.add_clause([[-d, -vert[-1]] for d in vert[:-1]])
	# no consec. dissonances within a single voice
	for voice in is_dissonance: problem.add_clause([[-a, -b] for a,b in context(voice, 2)])

	# 3th species, basically allows little else other than connective tissue
	if BEAT_TUPLE > 1:
		[ornaments_stepwise(problem, voice) for voice in voices]
		[from_ornament_to_downbeat_stepwise(problem, voice) for voice in voices]
		#[ornaments_diatonic(problem, voice) for voice in voices]
		[forbid_consencutive_dissonances(problem, a, b) for a,b in combinations(voices, 2)]
		[forbid_parallel_motion(problem, beat_notes(a), beat_notes(b)) for a,b in combinations(voices, 2)]

	return harmonies, is_dissonance

##############
# Quality control
##############

# normally stepwise motion is encouraged because it allows dissonance
# but occasionally a solution might occur that uses just broken chords
# in that case it is useful to force some stepwise motion to get a more dynamic solution
def minimal_stepwise_count(problem, voice, n):
	is_stepwise = [problem.make_var() for _ in range(len(voice)-1)]
	for a,b,v in zip(voice, voice[1:], is_stepwise):
		enforce_interval(problem, a, b, [-5,-2,2,5], [-v])
	problem.atleast(n, is_stepwise)
	return is_stepwise

# many complicated contrapuntal forms can be solved by making the whole dux a single pitch
# by simply requiring some minimum amount of different pitches in your dux you'll get actual solutions
def minimal_downbeat_variation(problem, voice, n):
	h = [problem.make_var() for _ in COFS]
	for p in COFS:
		problem.add_clause([[ h[p], -note[p]] for note in beat_notes(voice)])
		problem.add_clause([-h[p]] + [note[p] for note in beat_notes(voice)])

	problem.atleast(n, h)
	return h

def maximum_rests(problem, voice, n):
	problem.atmost(n, [note['rest'] for note in voice])

def maximum_note_length(problem, voice, n):
	is_different = [problem.make_var() for _ in range(len(voice)-1)]
	for a,b,v in zip(voice, voice[1:], is_different):
		enforce_interval(problem, a, b, [c for c in COFS if not c % 12 == 0], [-v])

	for vs in context(is_different, n):
		problem.atleast(1, vs)

	return is_different

# progression in CoF, first value 0, progression is relative to first value
def has_progression(problem, pitches, progression, n = 1, allow_edge = True):
	s = []
	for segment in context(pitches + pitches[:len(progression)]*allow_edge, len(progression)):
		s.append(problem.make_var())
		problem.add_clause([-s[-1], -segment[0]['rest']])

		for c in segment[0].keys():
			if c != 'rest':
				clauses = [[-segment[0][c], -s[-1], p[c+q]] if q+c in p else [-segment[0][c], -s[-1]] for p,q in zip(segment[1:], progression[1:])]
				problem.add_clause(clauses)

	problem.add_clause(s)
	#problem.atleast(n, s)
	return s

#########
# Generating output
#########

def getprop(solution, bitdict): return [k for k,v in bitdict.items() if v is None or v in solution][0]
def getindex(solution, bitlist): return [k for k,v in enumerate(bitlist) if v is None or v in solution][0]
def parse_voice(solution, voice): return [getprop(solution, note) if getprop(solution, note) != 'rest' else 100 for note in voice]

NOTE_NAMES = {
	-7: 'Cb',-6: 'Gb',-5: 'Db',-4: 'Ab',-3: 'Eb',-2: 'Bb', -1: 'F',
	 0: 'C',  1: 'G',  2: 'D',  3: 'A',  4: 'E',  5: 'B',   6: 'F#',
	 7: 'C#', 8: 'G#', 9: 'D#',10: 'A#',11: 'E#',12: 'B#', 13: 'Fx'}

INTERVAL_NAMES = {
	-7: 'd1',-6: 'd5',-5: 'm2',-4: 'm6',-3: 'm3',-2: 'm7',-1: 'P4', 
	 0: 'P1', 1: 'P5', 2: 'M2', 3: 'M6', 4: 'M3', 5: 'M7', 6: 'A4', 
	 7: 'A1', 8: 'A5', 9: 'A2',10: 'A6',11: 'A3',12: 'A7',
	}

ACC_NAMES = {-2:'𝄫', -1:'♭',0:' ',1:'♯', 2:'𝄪'}

class Solution:
	def __init__(self, solution, voices):
		self.voices = [(getprop(solution, tr), getprop(solution, ti), parse_voice(solution, voice)) for tr,ti,voice in voices]

	def __str__(self):
		s = ''
		for tr,ti,voice in self.voices:
			notes = ''.join(['cdefgab'[cof_to_diatonic(p)] + ACC_NAMES[cof_to_accidentals(p)] if p < 100 else '  ' for p in voice])
			s = f'{s}{"  -"[tr[1]]}{INTERVAL_NAMES[tr[0]]} {ti[0]:2}*{ti[1]:2} {"|".join(groupn(notes, MEASURE_TUPLE*BEAT_TUPLE*2))}\n'
		return s

def export_to_xml(voices):
	def assign_octaves(voice, pitch_center = 0):
		def near(p, q): return p%7 + 7*(q//7 + (q%7 - p%7 >= 4) - (q%7 - p%7 <=-4))
		voice = [cof_to_spelling(note) if note < 100 else None for note in voice]
		for i in range(len(voice)):
			a,b = voice[i-1:i+1] if i else ((pitch_center, 0), voice[0])
			voice[i] = (near(b[0], a[0] if a and (b[0] - a[0]) % 7 in [0,1,6] else pitch_center), b[1]) if b else None
		return voice			
	
	score = stream.Score(id='score')
	voices = [assign_octaves(voice, 63) for voice in voices]

	for i, v in enumerate(voices):
		part = stream.Part(id=f'part{i}')
		center = sorted([note[0] for note in v if note])[len([x for x in v if x])//2]
		if center <= 53: part.append(clef.BassClef())
		elif center <= 60: part.append(clef.Treble8vbClef())
		elif center >  60: part.append(clef.TrebleClef())

		if MEASURE_TUPLE == 2: part.append(meter.TimeSignature('2/2'))
		if MEASURE_TUPLE == 3: part.append(meter.TimeSignature('3/2'))

		part.append(instrument.Vocalist())
		pd, pa = -1,-2

		BEAT_LENGTH = 2 if BEAT_TUPLE == 1 else 1

		for i,n in enumerate(v):
			degree,acc = n if n else (None, None)
			if (degree, acc) == (pd, pa):
				part[-1].duration = duration.Duration(part[-1].duration.quarterLength + BEAT_LENGTH )
			else:
				if degree:
					part.append(note.Note(diatonic_to_chromatic(degree)-36-12*(i==len(v)-1), accidental=acc, duration=duration.Duration(BEAT_LENGTH)))
				else:
					part.append(note.Rest(duration=duration.Duration(BEAT_LENGTH)))

			pd, pa = degree, acc

		score.insert(0, part)

	return musicxml.m21ToXml.GeneralObjectExporter(score).parse().decode('utf-8').strip()


#######
# A specific canonic problem
#######
def square_canon():
	problem = Problem()
	M = MEASURE_TUPLE*BEAT_TUPLE
	N = 16*M

	dux = create_dux(problem, N)
	problem.add_clause([dux[0][0]])

	ordering = [12,8,4,0, 13,9,5,1, 14,10,6,2, 15,11,7,3]
	ordering = [y for x in ordering for y in range(x*M, (x+1)*M)]

	DIATONIC = PitchTransform(list(range(7)), 7, [1])
	rectus   = transformational_equivalence(problem, dux, PitchTransform([0], 7, [1]), TimeTransform([0], [1]), 0, (2*MEASURE_TUPLE*BEAT_TUPLE)) 
	retro    = transformational_equivalence(problem, dux, DIATONIC, TimeTransform(range(0,N,1), [-1]), 0, (2*MEASURE_TUPLE*BEAT_TUPLE))
	derectus = transformational_equivalence(problem, dux, DIATONIC, TimeTransform(range(0,N,1), [1], [ordering]), 0, (2*MEASURE_TUPLE*BEAT_TUPLE))
	deretro  = transformational_equivalence(problem, dux, DIATONIC, TimeTransform(range(0,N,1), [-1], [ordering]), 0, (2*MEASURE_TUPLE*BEAT_TUPLE))

	voices = [rectus, retro, derectus, deretro]
	harmony, is_dissonance = follow_rules_of_counterpoint(problem, [v[-1] for v in voices])
	maximum_rests(problem, dux, 0)
	maximum_note_length(problem, dux, 4)
	minimal_downbeat_variation(problem, dux, 5)

	for chord in harmony:
		problem.add_clause([-chord.root['rest']])

	for chord in harmony:
		for p in set(chord.root.keys()) - {'rest'}:
			problem.add_clause([-chord.root[p], chord.chroma[p]])

	maximum_note_length(problem, [chord.root for chord in harmony[:-1]], 2)

	for i,model in enumerate(problem.solutions_cpsat([x for note in dux for x in note.values()])):
		transforms = []
		for voice in [retro, derectus, deretro]:
			trans = [k for k,v in voice[0].items() if model[v-1] > 0][0][0]
			timed = [k for k,v in voice[1].items() if model[v-1] > 0][0][0]
			transforms.append((trans, timed))

		print(i, *transforms)
		print(' '.join([NOTE_NAMES[x].ljust(2) if x < 100 else '- ' for x in parse_voice(model, dux)]))
		print(' '.join([NOTE_NAMES[x].ljust(2) if x < 100 else '- ' for x in parse_voice(model, [chord.root for chord in harmony[:-1]])]))
		print(' '.join(['M ' if x else 'm ' for x in [model[chord.major-1] > 0 for chord in harmony[:-1]]]))

		for x in is_dissonance:
			markings = ['#' if model[d-1] > 0 else ' ' for i,d in enumerate(x)]
			markings = '|'.join([''.join(x) for x in groupn(markings, 4 if MEASURE_TUPLE == 2 else 3)]) + '||'
			print(markings)
		print()


		with open(f'square/solution_{i:04}.musicxml', 'w') as musicfile:
			musicfile.write(export_to_xml([x[-1] for x in Solution(model, voices).voices]))

# n is the amount of measures in the dux
# multi is how many simultanious such canons are present
def multi_modulation_canon(n, multi):
	problem = Problem()
	M = MEASURE_TUPLE*BEAT_TUPLE
	N = n*M

	mod = 4 - (MEASURE_TUPLE==3)

	duxs, voices = [], []
	for _ in range(multi):
		dux = create_dux(problem, N)
		problem.add_clause([dux[0][0]])
		maximum_rests(problem, dux, 3)
		#maximum_note_length(problem, dux, MEASURE_TUPLE+1)
		duxs.append(dux)

		dims = []
		for i in range(MEASURE_TUPLE):
			dims.append(transformational_equivalence(problem, dux, PitchTransform([i*mod], 12, [1]), TimeTransform([0], [1]), mod, (2*MEASURE_TUPLE*BEAT_TUPLE) * (i == MEASURE_TUPLE-1)))
		
		dim = [dims[0][0], dims[0][1], sum([x[2] for x in dims], start=[])]
		aug = transformational_equivalence(problem, dux, PitchTransform(list(range(12)), 12, [-1]), TimeTransform(list(range(0, N, MEASURE_TUPLE)), [MEASURE_TUPLE]), mod, (2*MEASURE_TUPLE*BEAT_TUPLE))
		
		voices += [dim, aug]

	voices = voices[::2] + voices[1::2]
	# fix a specific voice
	# for i,p in enumerate([0,1,-1,0, 1,5,0,1, 2,0,5,1]):
	# 	problem.add_clause([duxs[-1][i][p]])

	is_dissonance  = [are_beats_valid_dissonances(problem, voice) for voice in [v[-1] for v in voices]]
	annotated_beats = [[(note, i == len(voices)-1, d) for note,d in zip(beat_notes(voice), is_dissonance[i]) ] for i,voice in enumerate([v[-1] for v in voices])]
	harmonies = [harmonic_vector(problem, harmony) for harmony in groupn(list(zip(*annotated_beats)), MEASURE_TUPLE)]
	[forbid_parallel_motion(problem, a, b) for a,b in combinations([v[-1] for v in voices], 2)]

	for chord in harmonies:
		problem.add_clause([-chord.root['rest']])

	for chord in harmonies:
		for p in set(chord.root.keys()) - {'rest'}:
			problem.add_clause([-chord.root[p], chord.chroma[p]])

	for i,model in enumerate(problem.solutions_cpsat([x for dux in duxs for note in dux for x in note.values()])):
		aug_dux1_transposition = [k for k,v in aug[0].items() if model[v-1] > 0]
		aug_dux1_time = [k for k,v in aug[1].items() if model[v-1] > 0]

		print(i, aug_dux1_transposition[0][0], aug_dux1_time[0][0])
		print(' '.join([NOTE_NAMES[x].ljust(2) if x < 100 else '- ' for x in parse_voice(model, sum(duxs, start=[]))]))
		print(' '.join([NOTE_NAMES[x].ljust(2) if x < 100 else '- ' for x in parse_voice(model, [chord.root for chord in harmonies[:-1]])]))
		print(' '.join(['M ' if x else 'm ' for x in [model[chord.major-1] > 0 for chord in harmonies[:-1]]]))

		for x in is_dissonance:
			markings = ['#' if model[d-1] > 0 else ' ' for i,d in enumerate(x)]
			markings = '|'.join([''.join(x) for x in groupn(markings, 4 if MEASURE_TUPLE == 2 else 3)]) + '||'
			print(markings)
		print()

		with open(f'out/solution_{i:04}.musicxml', 'w') as musicfile:
			musicfile.write(export_to_xml([x[-1] for x in Solution(model, voices).voices]))

def augmentation_canon(n):
	problem = Problem()
	M = MEASURE_TUPLE*BEAT_TUPLE
	N = n*M

	dux = create_dux(problem, N)
	problem.add_clause([dux[0][0]])
	maximum_rests(problem, dux, 0)
	maximum_note_length(problem, dux, MEASURE_TUPLE+1)

	voices = []
	for aug_factor in [1,2,3]:
		pitch = PitchTransform(list(range(12)), 12, [1,-1]) if aug_factor else PitchTransform([0], 12, [1])
		voices.append(transformational_equivalence(problem, dux, pitch, TimeTransform([0], [aug_factor]), 0, 6*N-aug_factor*N + (2*MEASURE_TUPLE*BEAT_TUPLE)))

	is_dissonance = [are_beats_valid_dissonances(problem, voice) for voice in [v[-1] for v in voices]]
	annotated_beats = [[(note, i == len(voices)-1, d) for note,d in zip(beat_notes(voice), is_dissonance[i]) ] for i,voice in enumerate([v[-1] for v in voices])]
	harmonies = [harmonic_vector(problem, harmony) for harmony in groupn(list(zip(*annotated_beats)), MEASURE_TUPLE)]
	[forbid_parallel_motion(problem, a, b) for a,b in combinations([v[-1] for v in voices], 2)]

	for chord in harmonies:
		problem.add_clause([-chord.root['rest']])

	for chord in harmonies:
		for p in set(chord.root.keys()) - {'rest'}:
			problem.add_clause([-chord.root[p], chord.chroma[p]])

	for i,model in enumerate(problem.solutions_cpsat([x for note in dux for x in note.values()])):
		print(i)
		print(' '.join([NOTE_NAMES[x].ljust(2) if x < 100 else '- ' for x in parse_voice(model, dux)]))
		print(' '.join([NOTE_NAMES[x].ljust(2) if x < 100 else '- ' for x in parse_voice(model, [chord.root for chord in harmonies[:-1]])]))
		print(' '.join(['M ' if x else 'm ' for x in [model[chord.major-1] > 0 for chord in harmonies[:-1]]]))

		for x in is_dissonance:
			markings = ['#' if model[d-1] > 0 else ' ' for i,d in enumerate(x)]
			markings = '|'.join([''.join(x) for x in groupn(markings, 4 if MEASURE_TUPLE == 2 else 3)]) + '||'
			print(markings)
		print()

		with open(f'aug/solution_{i:04}.musicxml', 'w') as musicfile:
			musicfile.write(export_to_xml([x[-1] for x in Solution(model, voices).voices]))

# multi_modulation_canon(6, 2)
# square_canon()
augmentation_canon(6)
