import sys, re
from itertools import *
# Kies het octaaf zo dat deze regels gelden voor alle stemmen
# - totale bereik mag niet meer zijn dan octaaf + 1
# - geen sprongen van een septiem of meer
# - een sextsprong moet een tegenbeweging hebben
# - een kwintsprong mag alleen door in dezelfde richting gevolgd door stap of kwartsprong
# - een kwartsprong mag alleen door in dezelfde beweging gevolgd door een kwint, stap of terts.
# - een tertssprong mag alleen door in dezelfde beweging gevolgd door een stap, terts of kwart.
# - een secunde mag worden gevolgd door elke interval
# - na 2 sprongen moet er altijd tegenbeweging plaatsvinden


def zerofy(path): return tuple([x-path[0] for x in path])
def tangle(path):
	out = []
	for column in range(4):
		for row in range(4):
			out += path[(row*4+column)*2:(row*4+column)*2+2]
	return out

def reverse(path): return path[::-1]
def translate(path, n): return path[n:] + path[:n]

def transform(voice, transpose, invert, time, retrograde, derectus):
	if derectus: voice = tangle(voice)
	if retrograde: voice = reverse(voice)
	voice = [x + transpose if type(x) is int else (x[0] + transpose, *x[1:]) for x in voice]
	if invert: voice = [(6 - x[0], *x[1:]) for x in voice]
	voice = translate(voice, time)
	return voice


def compact(x, y):
	while abs(x-y) > 3: y += 7 if x > y else -7
	return y

def goup(x, y):
	y -= 70
	while y < x: y += 7
	return y

def godown(x, y):
	y += 70
	while y > x: y -= 7
	return y

def find_path(done, todo, allow_leaps):
	#print(done, todo)
	if done and max(done) - min(done) > int(sys.argv[1]):
		return

	if not done:
		for x in find_path(done + todo[:1], todo[1:], allow_leaps):
			yield x
		return
	elif not todo:
		if abs(done[0] - done[-1]) <= 4+allow_leaps:
			yield done
	else:
		d = (todo[0] - done[-1]) % 7
		
		if (d <= 2-allow_leaps or d >= 5+allow_leaps) and (d != 0 or not allow_leaps):
			for x in find_path(done + [compact(done[-1], todo[0])], todo[1:], allow_leaps):
				yield x
			return
		else:
			for x in find_path(done +  [goup(done[-1], todo[0])], todo[1:], allow_leaps):
				yield x
			for x in find_path(done +[godown(done[-1], todo[0])], todo[1:], allow_leaps):
				yield x

def guess_tonic(pitches):

	vector = [0]*7
	for i,j in combinations(pitches, 2):
		interval = max(i, j) - min(i, j)
		if interval == 4: vector[min(i, j)%7] += 32
		if interval == 3: vector[max(i, j)%7] += 32
		if interval == 2:
			vector[min(i, j)%7] += 2
			vector[(min(i, j)-2)%7] += 1
		if interval == 5:
			vector[max(i, j)%7] += 2
			vector[(max(i, j)-2)%7] += 1

	#m = min(pitches)
	#vector[m%7] += 3
	#vector[(m-2)%7] += 2
	#for p in pitches:
	#	vector[p%7] += 3
	#	vector[(p-2)%7] += 2
	#	vector[(p-4)%7] += 1

	return vector.index(max(vector))

def get_chords(voices):
	chords = []
	measures = list(zip(*[v[-1] for v in voices]))
	for strong, weak in zip(measures[::2], measures[1::2]):
		consonants = [note[0] for note in list(strong) + list(weak) if not note[2]]
		chords.append(guess_tonic(consonants))
	return chords

progressions = set([tuple([x+r for x in p]) for p in product([0,2,5], [1,3,0,2,5], [4,6], [0,2,5]) for r in range(7)])
def get_functionality_score(voices):
	chords = get_chords(voices)
	return sum((tuple(chords[r:] + chords[:r])[:4]) in progressions for r in range(len(chords))), chords

def parse_solution(text):
	VOICE_COUNT = 4 #text[:text.index('\n\n')].count('\n')//2
	OFFSET = len(re.findall(r'^ *[-][ \d]+m *', text)[0])

	lines = text.split('\n')

	voices = lines[0:2*VOICE_COUNT:2]
	dissonances = lines[1:2*VOICE_COUNT:2]
	#harmony = lines[2*VOICE_COUNT] dont care
	variables = {name:float(value) for name, value in re.findall(r'([\w_]+) *= ([\d.]+)',text)}

	parsed_voices = []
	for voice, row in zip(voices, dissonances):
		row = [(d[0] if d[0] != ' ' else None, d[1] == '+') for d in re.findall('..', row[OFFSET:])]

		transposition = 0 if voice[0] == ' ' else int(voice[0]) - 1
		time_offset = -int(re.findall(r'\d+m', voice)[0][:-1])
		notes = re.findall('..', voice[OFFSET:])
		notes = [('cdefgab'.index(note[0]), '♭ ♯'.index(note[1]) - 1, *dissonance) for note, dissonance in zip(notes, row)]
		parsed_voices.append((transposition, time_offset, notes))
	voices = parsed_voices

	return (voices, variables)

from music21 import *


def diatonic_to_chromatic(d): return d//7*12 + [0,2,4,5,7,9,11][d%7]

def export_to_xml(voices):
	clefs = [clef.TrebleClef(), clef.TrebleClef(), clef.Treble8vbClef(), clef.BassClef()]
	instruments = [instrument.Vocalist(), instrument.Vocalist(), instrument.Vocalist(), instrument.Vocalist()]

	score = stream.Score(id='score')

	for i, c, instr, v in zip(range(len(voices)), clefs, instruments, voices):
		part = stream.Part(id=f'part{i}')
		part.append(c)
		part.append(instr)
		pd, pa = -1,-2

		for degree,acc,_,_ in v[-1]:
			if (degree, acc) == (pd, pa):
				part[-1].duration = duration.Duration(part[-1].duration.quarterLength + 2)
			else:
				part.append(note.Note(diatonic_to_chromatic(degree), accidental=acc, duration=duration.Duration(2)))

			pd, pa = degree, acc

		score.insert(0, part)

	GEX = musicxml.m21ToXml.GeneralObjectExporter(score)
	return GEX.parse().decode('utf-8').strip()


def chunks(itr, n):
	buffer = []
	for line in itr:
		buffer.append(line)
		if len(buffer) == n:
			yield buffer
			buffer = []

SOLUTION_SIZE = 12
written = 0
best_so_far = -1

def all_multi_valid_paths(voices, leap):
	rectus_paths = set(zerofy(path) for path in find_path([], [x[0] for x in voices[0][2]], leap))
	for path in rectus_paths:
		rectus   = [(pitch, *note[1:]) for note, pitch in zip(voices[0][2], path)]
		derectus = [x[0] for x in transform(rectus, voices[2][0], False, voices[2][1], False, True)]
		if max(derectus) - min(derectus) <= int(sys.argv[1]):
			yield path

for i, chunk in enumerate(chunks(sys.stdin, SOLUTION_SIZE)):
	if i % 10000 == 0:
		print(i)
	voices, variables = parse_solution(''.join(chunk))
	#if variables['note_count'] < 46:
	#	continue

	dissonance_count = [d for v in voices for _,_,d,_ in v[2] if d == 'S']

	paths = list(all_multi_valid_paths(voices, 0))
	if not paths:
		paths = list(all_multi_valid_paths(voices, 1))
		if not paths:
			paths = list(all_multi_valid_paths(voices, 2))

	# if so reapply transformations of all voices to rectus with correct octaves to get a true solution
	if paths:
		best = sorted(paths, key=lambda x: max(x) - min(x))[0]
		t = voices[0][2][0][0]
		octave = 7*6

		rectus   = [(pitch+t, *note[1:]) for note, pitch in zip(voices[0][2], best)]
		retro    = transform(rectus, voices[1][0], True , -voices[1][1], True, False)
		derectus = transform(rectus, voices[2][0], False, -voices[2][1], False, True)
		deretro  = transform(rectus, voices[3][0], True , -voices[3][1], True,  True)
		voices = [(*x[:2], [(note[0]+t, *original[1:]) for original, note in zip(x[2], y)]) for x,y,t in zip(voices, [rectus, retro, derectus, deretro], [octave, octave-7, octave-14, octave-14])]

		fscore, chords = get_functionality_score(voices)
		score = variables['note_count'] * len(dissonance_count) * fscore**2
		#if score <= best_so_far:
		#	continue

		print('         ' + '   '.join(['CDEFGAB'[x] for x in chords]), fscore)
		print(''.join(chunk))

		# finally we can write to xml
		print("WRITING", fscore)
		f = open(f'../test_file_{written:2}.musicxml', 'w')
		f.write(export_to_xml(voices))
		f.close()
		written += 1
		best_so_far = score
		if written == 20:
			break

