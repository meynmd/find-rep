from __future__ import division
from collections import defaultdict
from copy import deepcopy
import sys
import math
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


def find_compactness(pattern, score):
    pattern = pattern[0]
    first, last = min(desc[1] for desc in pattern), max(desc[1] for desc in pattern)
    low, high = min(desc[0] for desc in pattern), max(desc[0] for desc in pattern)
    timespan_notes = [n for n in score.flat.notes if n.offset >= first and n.offset <= last]
    region_notes = []
    for n in timespan_notes:
        include = True
        for p in n.pitches:
            if p.diatonicNoteNum < low or p.diatonicNoteNum > high:
                include = False
        if include:
            region_notes.append(n)

    return float(len(pattern)) / len(region_notes)


def find_cov_score(pattern, score):
    pattern = pattern[0]
    first, last = min(desc[1] for desc in pattern), max(desc[1] for desc in pattern)
    low, high = min(desc[0] for desc in pattern), max(desc[0] for desc in pattern)
    timespan_notes = [n for n in score.flat.notes if n.offset >= first and n.offset <= last]

    region_notes = timespan_notes

    # region_notes = []
    # for n in timespan_notes:
    #     include = True
    #     for p in n.pitches:
    #         if p.diatonicNoteNum < low or p.diatonicNoteNum > high:
    #             include = False
    #     if include:
    #         region_notes.append(n)

    cov_score = min(1., float(len(pattern)) / len(region_notes))
    score_time = max(n.offset for n in score.flat.notes)
    comp_score =  (last - first) / float(score_time)

    return cov_score - comp_score


def get_mtps(score):
    notes = []
    tup2note = {}
    for n in score.flat.notes:
        if isinstance( n, music21.note.Note ):
            descriptor = (n.pitch.diatonicNoteNum, n.offset, n)
            notes.append( descriptor )
            tup2note[descriptor] = n

    vec2points = make_vec_table( notes, [1., 1., 0] )
    pv_dict = extract_patterns( vec2points )
    pv_items = [(p, v) for p, v in pv_dict.items() if len( p ) > 3 and len( p ) < 20]
    pv_items = [(p, v) for p, v in pv_items if max( n[1] for n in p ) != min( n[1] for n in p )]

    # combined_score = [find_cov_score( pv, score ) for pv in pv_items]
    # occur_score = [math.log( len( p ) ) * len( v ) ** 2 / float( len( score.flat.notes ) ** 2 ) for p, v in pv_items]
    # idxs = sorted( [i for i in range( len( pv_items ) )], key=lambda x: combined_score[x], reverse=True )
    pv_items = sorted(pv_items, key=lambda x : find_cov_score(x, score))
    if len(pv_items) > 2000:
        pv_items = pv_items[:2000]
    return pv_items



if __name__ == '__main__':
    score = music21.converter.parse( 'bf1i.krn' )
    for pattern, v in get_mtps(score)[:20]:
        notes = sorted([(pitch, dur, name) for (pitch, dur, name) in pattern], key=lambda x : x[1])
        print '*' * 80
        print sorted(pattern, key=lambda x : x[1])
        print v
        print '\n' + '*' * 80

        for n in score.flat.notes:
            n.style.color = None
        pattern_notes = {n : (p, d) for p, d, n in pattern}
        for note in score.flat.notes:
            if note in pattern_notes and note.pitch.diatonicNoteNum == pattern_notes[note][0] and note.offset == pattern_notes[note][1]:
                note.style.color = 'blue'

        t = ''
        for n in sorted(pattern, key=lambda x : x[1]):
            t += '{}, {}\n'.format(n[2].pitch, n[1])

        score.metadata = music21.metadata.Metadata(movementName=t, composer=v)
        score.show()


    """
    notes = []
    tup2note = {}
    for n in score.flat.notes:
        if isinstance( n, music21.note.Note ):
            descriptor = (n.pitch.diatonicNoteNum, n.offset, n)
            notes.append(descriptor)
            tup2note[descriptor] = n
    
    vec2points = make_vec_table( notes, [1., 1., 0] )
    pv_dict = extract_patterns( vec2points )
    pv_items = [(p, v) for p, v in pv_dict.items() if len(p) > 3 and len(p) < 20]
    pv_items = [(p, v) for p, v in pv_items if max(n[1] for n in p) != min(n[1] for n in p)]
    
    # occur_score = [len(p) * len(v) / float(len(score.flat.notes)) for p, v in pv_items]
    # idxs = sorted([i for i in range(len(pv_items))], key=lambda x : occur_score[x], reverse=True)
    
    # combined_score = {idx : find_combined_score(pv_items[idx], score) for idx in idxs}
    # idxs = sorted(idxs[:1000], key=lambda x : 0.95 * combined_score[x] + 0.05 * occur_score[x], reverse=True)
    # sbs = [(idx, pv_items[idx]) for idx in idxs]
    
    # pv_items = pv_items[:1000]
    combined_score = [find_cov_score(pv, score) for pv in pv_items]
    occur_score = [math.log(len(p)) * len(v)**2 / float(len(score.flat.notes)**2) for p, v in pv_items]
    idxs = sorted([i for i in range(len(pv_items))], key=lambda x : combined_score[x], reverse=True)
    
    # for idx in idxs[:50]:
    #     pattern = pv_items[idx]
    #     print '*' * 80 + '\nscore {}:'.format( combined_score[idx] )
    #     notes = sorted([(pitch, dur, name) for (pitch, dur, name) in pattern[0]], key=lambda x : x[1])
    #     print sorted(pattern[0], key=lambda x : x[1])
    #     print pattern[1]
    #     print '\n' + '*' * 80
    # 
    #     for n in score.flat.notes:
    #         n.style.color = None
    #     pattern_notes = {n : (p, d) for p, d, n in pattern[0]}
    #     for note in score.flat.notes:
    #         if note in pattern_notes and note.pitch.diatonicNoteNum == pattern_notes[note][0] and note.offset == pattern_notes[note][1]:
    #             note.style.color = 'blue'
    # 
    #     t = ''
    #     for n in sorted(pattern[0], key=lambda x : x[1]):
    #         t += '{}, {}\n'.format(n[2].pitch, n[1])
    # 
    #     score.metadata = music21.metadata.Metadata(movementName=t, composer=pattern[1])
    #     score.show()
    """