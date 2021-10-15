# MusicSAT

Solves complicated canonic forms using SAT.

This is just something for personal use but feel free to aks questions. To use you need at least python 3.7 and pysat (https://pysathq.github.io/), if you want to generate musicxml files you also need music21 (https://web.mit.edu/music21/).

The first two dozen lines in solve_for_time() in cnf.py essentially define the canonic form, all the other code is generic for any contrapuntal composition. melody_octaves.py is a tool that converts the output of cnf.py to musicxml files, but it's a very ad-hoc tool.

