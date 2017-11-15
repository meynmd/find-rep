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
pv_dict = extract_patterns( vec2points )
mtps = [sorted(list(set((n[0], n[1], n[2].pitch) for n in pattern)), key=lambda x : x[1]) for pattern in pv_dict if len(pattern) > 2]
compact_score = [(max([y[1] for y in x]) - min([y[1] for y in x])) / len(x) for x in mtps]
occur_score = [len(p) * len(v) for p, v in pv_dict.items()]

sorted_by_compact = sorted( mtps, key=lambda x : (max([y[1] for y in x]) - min([y[1] for y in x])) / len(x) )

idxs = sorted([i for i in range(len(mtps))], key=lambda x :  occur_score[x])
sbs = [(idx, mtps[idx]) for idx in idxs if mtps[idx][0][1] < 2.]



for idx, pattern in sbs[:50]:
    print '*' * 80 + '\nPattern, score {}*{}:'.format(compact_score[idx], occur_score[idx])
    pattern = sorted([(pitch, dur, name) for (pitch, dur, name) in pattern], key=lambda x : x[1])
    for n in pattern:
        print n
    print '\n' + '*' * 80





