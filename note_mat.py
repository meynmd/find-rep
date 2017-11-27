from __future__ import division
from math import *
from collections import defaultdict
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import music21


class NoteMatrix:
    pad_pitch = 0
    pad_dur = 0

    def __init__( self, score, timestep_min=None ):
        self.music_score = score
        scorenotes = [n for n in score.flat.notes]
        self.timestep = min( n.duration.quarterLength for n in scorenotes )
        if timestep_min is not None:
            self.timestep = timestep_min
        self.pitch_max = max( max( p.midi for p in sn.pitches ) for sn in scorenotes ) + NoteMatrix.pad_pitch
        self.pitch_min = min( min( p.midi for p in sn.pitches ) for sn in scorenotes ) - NoteMatrix.pad_pitch
        self.pitch_range = self.pitch_max - self.pitch_min
        self.mat = np.zeros( [self.pitch_range + 1, self.ql2ts( score.duration.quarterLength )] )
        self.loc2point = {}
        self.populate( scorenotes )
        self.vec_table = self.make_vec_table()
        self.mtp_table = self.extract_patterns()

    def make_kernel(self, pattern, sigma, pad_p=0, pad_d=0):
        low, high, first, last = self.find_bounds(pattern)
        kernel = np.zeros([high - low + 1, last - first + 1])
        for p, o in pattern:
            kernel[p - low, o - first] = 1.
        kernel = np.pad(kernel, ((pad_p, pad_p), (pad_d, pad_d)), 'constant')
        return gaussian_filter( kernel, sigma )

    def populate( self, notes ):
        for point in notes:
            if isinstance( point, music21.note.Rest ):
                continue
            elif isinstance( point, music21.chord.Chord ):
                for pitch in point.pitches:
                    self.write_point(point, 1.)
            else:
                self.write_point(point, 1.)

    def write_point(self, point, value):
        locations = [(self.rel_pitch( p.midi ), self.ql2ts( point.offset )) for p in point.pitches]
        for location in locations:
            self.mat[location] = value
            self.loc2point[location] = point

    def ql2ts( self, ql ):
        return int( ceil( ql / self.timestep ) )

    def ts2ql( self, ts ):
        return ts * self.timestep

    def rel_pitch( self, midi ):
        return midi - self.pitch_min

    def abs_pitch( self, rel ):
        return self.pitch_min + rel

    def make_vec_table(self):
        keys = sorted(self.loc2point.keys())
        vec_points = defaultdict( list )
        for i, row in enumerate( keys ):
            for j, col in enumerate( keys ):
                if i <= j:
                    continue
                v = tuple( [row[k] - col[k] for k in range( len( row ) )] )
                vec_points[v].append( col )
        return vec_points

    def extract_patterns(self):
        pattern_vec = defaultdict( list )
        for vec, pointset in self.vec_table.items():
            pattern_vec[tuple( sorted( pointset, key=lambda x : (x[1], x[0]) ) )].append( vec )
        return pattern_vec

    def find_bounds(self, pattern):
        by_pitch, by_offset = sorted(pattern, key=lambda x : x[0]), sorted(pattern, key=lambda x : x[1])
        low, high, first, last = by_pitch[0][0], by_pitch[-1][0], by_offset[0][1], by_offset[-1][1]
        return low, high, first, last

    def find_compact(self, pattern, vec):
        low, high, first, last = self.find_bounds(pattern)
        timespan_points = [(p, o) for p, o in self.loc2point if first <= o <= last]
        bound_box_points = [(p, o) for p, o in timespan_points if low <= p <= high]
        # return float(len(pattern)) / len(timespan_points)
        return float(len(pattern)) / len(bound_box_points)

    def find_coverage( self, pattern, vec ):
        pattern_points = pattern + tuple(list(set([(p + vp, o + vo) for p, o in pattern for vp, vo in vec])))
        return float(len(pattern_points)) / len(self.loc2point)

    def get_mtps(self):
        return self.mtp_table.keys()

    def get_sorted_mtps(self, min_length=3):
        mtps = [(p, v) for p, v in self.mtp_table.items() if len(p) > min_length]
        compact_score = [self.find_compact(p, v) for p, v in mtps]
        coverage_score = [self.find_coverage(p, v) for p, v in mtps]
        idxs = sorted([i for i in range(len(mtps))],
                      key=lambda x : compact_score[x] * coverage_score[x],
                      reverse=True)
        return [(mtps[idx][0], compact_score[idx], coverage_score[idx]) for idx in idxs]

        # return sorted(
        #     [p for p, v in self.mtp_table.items()],
        #     key=lambda x : self.find_compact(p, v) * self.find_coverage(p, v)
        # )