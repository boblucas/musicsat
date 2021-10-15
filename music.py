from itertools import chain

# this music theory is very specific to the SAT implementation

# in SAT we use a 1:many chroma -> diatonic mapping using only flats like this:
# C Db D Eb E F Gb G Ab A Bb B
def chroma_is_flat(c): return [0,1,0,1,0,0,1,0,1,0,1,0][c%12]
def dia_from_chroma(c): return [0,1,1,2,2,3,4,4,5,5,6,6][c%12] + c//12*7
def chromas_from_dia(d): return [(0,), (1,2), (3,4), (5,), (6,7), (8,9), (10, 11)][d]
def chroma_from_dia(d, is_flat): return chromas_from_dia(d)[0 if is_flat else -1]

# transformations on lists that corrospond to canonic time transforms
def rotate(V, r): return V[r:] + V[:r]
def columns(V, n, k): return list(chain(*[V[(j*n+i)*k:(j*n+i)*k+k] for i in range(n) for j in range(n)]))
def columns_rev(V, n, k): return list(chain(*[V[(j*n+i)*k:(j*n+i)*k+k] for i in range(n) for j in list(range(n))[::-1]]))

# excluding tritones
def is_aug_or_dim(i, j):
	for cof in range(-5, 0):
		chroma = ((cof-1) * 7) % 12
		for chroma_to in [((chroma+8)+i*7)%12 for i in range(-1, abs(cof) - 1)]:
			if (i, j) == (chroma, chroma_to) or (j, i) == (chroma, chroma_to):
				return True
	return False


