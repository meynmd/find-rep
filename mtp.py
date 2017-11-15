from __future__ import division
from collections import defaultdict
from copy import deepcopy
import music21


def make_vec_table( points, weights ):
    vec_point = defaultdict( list )
    keys = sorted( points )
    for i, row in enumerate( keys ):
        for j, col in enumerate( keys ):
            if i <= j:
                continue
            v = tuple( [row[k] - col[k] for k in range( len( row ) ) if weights[k] > 0.] )
            vec_point[v].append( col )
    return vec_point


def extract_patterns( vec_table ):
    pattern_vec = defaultdict( list )
    for vec, pattern in vec_table.items():
        pattern_vec[tuple( pattern )].append( vec )
    patterns = [list(set(p)) for p in pattern_vec]
    return pattern_vec


score = music21.converter.parse( 'bf1i.krn' )
notes = []
tup2note = {}
for n in score.flat.notes:
    if isinstance( n, music21.note.Note ):
        # descriptor = (ord( n.pitch.step ) - ord( 'A' ), n.offset, n)
        descriptor = (n.pitch.diatonicNoteNum, n.offset, n)
        notes.append(descriptor)
        tup2note[descriptor] = n

vec2points = make_vec_table( notes, [1., 1., 0] )
mtps = extract_patterns( vec2points )
mtps = [sorted(list(set((n[0], n[1], n[2].pitch) for n in pattern)), key=lambda x : x[1]) for pattern in mtps if len(pattern) > 2]
sorted_by_compact = sorted( mtps, key=lambda x : (max([y[1] for y in x]) - min([y[1] for y in x])) / len(x) )

for pattern in sorted_by_compact[:200]:
    print '*' * 80 + '\nPattern:'
    pattern = sorted([(pitch, dur, name) for (pitch, dur, name) in pattern], key=lambda x : x[1])
    for n in pattern:
        print n
    print '\n' + '*' * 80





