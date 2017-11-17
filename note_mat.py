from __future__ import division
from math import *
import sys
import numpy as np
import music21


class NoteMatrix:
    pad_pitch = 0
    pad_dur = 0

    def __init__( self, score, timestep_min=None ):
        scorenotes = [n for n in score.flat.notes]
        self.timestep = min( n.duration.quarterLength for n in scorenotes )
        if timestep_min is not None:
            self.timestep = timestep_min
        self.pitch_max = max( max( p.midi for p in sn.pitches ) for sn in scorenotes ) + NoteMatrix.pad_pitch
        self.pitch_min = min( min( p.midi for p in sn.pitches ) for sn in scorenotes ) - NoteMatrix.pad_pitch
        self.pitch_range = self.pitch_max - self.pitch_min
        self.mat = np.zeros( [self.pitch_range + 1, self.ql2ts( score.duration.quarterLength )] )
        self.populate( scorenotes )


    def populate( self, notes ):
        for point in notes:
            if isinstance( point, music21.note.Rest ):
                continue
            elif isinstance( point, music21.chord.Chord ):
                for pitch in point.pitches:
                    self.mat[self.rel_pitch( pitch.midi ),
                             self.ql2ts( point.offset )] = \
                        self.ql2ts( point.duration.quarterLength )
            else:
                self.mat[self.rel_pitch( point.pitch.midi ),
                         self.ql2ts( point.offset )] = \
                    point.duration.quarterLength

    def ql2ts( self, ql ):
        return int( ceil( ql / self.timestep ) )

    def ts2ql( self, ts ):
        return ts * self.timestep

    def rel_pitch( self, midi ):
        return midi - self.pitch_min

    def abs_pitch( self, rel ):
        return self.pitch_min + rel

