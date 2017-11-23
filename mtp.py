from __future__ import division
from collections import defaultdict
from copy import deepcopy
import sys
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


# def coverage(score, pattern_vec, vec_point):
#     for pat, vec in pattern_vec:

def find_compactness(pattern, score):
    first, last = min(desc[1] for desc in pattern), max(desc[1] for desc in pattern)
    # timespan_notes = score.flat.getElementsByOffset(first, last).getElementsByClass('Note')
    low, high = min(desc[0] for desc in pattern), max(desc[0] for desc in pattern)
    timespan_notes = [n for n in score.flat.notes if n.offset >= first and n.offset <= last]
    # region_notes = [n for n in timespan_notes
    #                 if low <= n.pitch.diatonicNoteNum <= high]
    region_notes = []
    for n in timespan_notes:
        include = True
        for p in n.pitches:
            if p.diatonicNoteNum < low or p.diatonicNoteNum > high:
                include = False
        if include:
            region_notes.append(n)


    return float(len(pattern)) / len(region_notes)


score = music21.converter.parse( 'bf1i.krn' )
notes = []
tup2note = {}
# score = score.measures(0,5)
for n in score.flat.notes:
    if isinstance( n, music21.note.Note ):
        # descriptor = (ord( n.pitch.step ) - ord( 'A' ), n.offset, n)
        descriptor = (n.pitch.diatonicNoteNum, n.offset, n)
        notes.append(descriptor)
        tup2note[descriptor] = n

vec2points = make_vec_table( notes, [1., 1., 0] )
pv_dict = extract_patterns( vec2points )
pv_dict = [(p, v) for p, v in pv_dict.items() if len(p) > 1]
pv_dict = [(p, v) for p, v in pv_dict if max(n[1] for n in p) != min(n[1] for n in p)]

# mtps = [sorted(list(set(((n[0], n[1], n[2].pitch), tuple(v)) for n in pattern)), key=lambda x : x[1]) for pattern, v in pv_dict]
# compact_score = [len(x) / (max([y[1] for y in x]) - min([y[1] for y in x])) for x, v in pv_dict]
compact_score = []
# for i, (p, v) in enumerate(pv_dict):
#     if i % 50 == 0:
#         print >> sys.stderr, '{0:2%}'.format(i / len(pv_dict)),
#     if i % 1000 == 0:
#         print >> sys.stderr
#     compact_score.append(find_compactness(p, score))
compact_score = [find_compactness(p, score) for p, v in pv_dict]
occur_score = [len(p) * len(v) / float(len(score.flat.notes)) for p, v in pv_dict]

# sorted_by_compact = sorted( mtps, key=lambda x : (max([y[1] for y in x]) - min([y[1] for y in x])) / len(x) )

idxs = sorted([i for i in range(len(pv_dict))], key=lambda x : compact_score[x] + occur_score[x], reverse=True)
sbs = [(idx, pv_dict[idx]) for idx in idxs]


for idx, pattern in sbs[:1000]:
    print '*' * 80 + '\nPattern, score {} * {}:'.format(compact_score[idx], occur_score[idx])
    notes = sorted([(pitch, dur, name) for (pitch, dur, name) in pattern[0]], key=lambda x : x[1])
    print sorted(pattern[0], key=lambda x : x[1])
    print pattern[1]
    print '\n' + '*' * 80

    # for n in score.flat.notes:
    #     n.style.color = None
    # pattern_notes = {n : (p, d) for p, d, n in pattern[0]}
    # for note in score.flat.notes:
    #     if note in pattern_notes and note.pitch.diatonicNoteNum == pattern_notes[note][0] and note.offset == pattern_notes[note][1]:
    #         note.style.color = 'blue'
    #
    # t = ''
    # for n in sorted(pattern[0], key=lambda x : x[1]):
    #     t += '{}, {}\n'.format(n[2].pitch, n[1])
    #
    # score.metadata = music21.metadata.Metadata(movementName=t, composer=pattern[1])
    # score.show()
