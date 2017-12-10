from __future__ import division
from math import *
from copy import deepcopy
import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn.functional as fn
from torch.autograd import Variable
import music21
from note_mat import *
from mtp import *

def plot_mat(tensor, notematrix):
    import matplotlib.pyplot as plt

    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    mat = tensor.numpy()
    fig, ax = plt.subplots()

    plt.imshow(np.flip(mat, 0), aspect='auto', extent=(0, mat.shape[1]*notematrix.timestep, notematrix.pitch_min, notematrix.pitch_max))
    ax.set_xticks(range(0, int(mat.shape[1]*notematrix.timestep), 4))

    plt.show()

def convolve_single_channel(matrix, kernel, pad):
    output = fn.conv2d(Variable(matrix.unsqueeze(0).unsqueeze(0)), Variable(kernel.unsqueeze(0).unsqueeze(0)), padding=pad)
    return output.data.squeeze()

def make_kernel(pattern, amp, pad, sig):
    k = amp * pattern
    k = np.pad(k, pad, 'constant')
    return gaussian_filter(k, sig)

def evaluate_patterns(filename, patterns):
    pass

def copy_pattern(score, pattern):
    stream = music21.stream.Stream()
    first_note, last_note = min([p[2] for p in pattern], key=lambda x : x.offset), min([p[2] for p in pattern], key=lambda x : x.offset)
    measure_numbers = (first_note.measureNumber, last_note.measureNumber)
    for n in score.recurse().measures(*measure_numbers).recurse().notes:
        stream.append(deepcopy(n))
    return stream

def do_convolutions( mtps, notematrix, sigma=1., pad=5 ):
    print 'doing convolutions...\n'
    activation_sums, kernels = [], []
    for i, pattern in enumerate(mtps):
        kernel = notematrix.make_kernel( pattern, sigma, 5, 5 )
        print 'pattern {}, matrix {}, kernel {}'.format(i, notematrix.mat.shape, kernel.shape), '\r'

        pad = (int( floor( kernel.shape[0] / 2 ) ), int( floor( kernel.shape[1] / 2 ) ))
        out = convolve_single_channel( torch.from_numpy( notematrix.mat ), torch.from_numpy( kernel ), pad )
        out = Variable( out.view( 1, out.size( 0 ), -1 ) )
        out = fn.max_pool2d( out, (1, int( 1. / notematrix.timestep )) )
        out = out.data.view( out.data.size()[1], -1 )
        arr = out.numpy()
        actsum = np.sum( arr**2 ) / float( len( pattern ) ) ** 2
        # actsum = np.sum( out.numpy() ) / float( len( pattern ) ) ** 2
        activation_sums.append( actsum )
        kernels.append( kernel )
    return activation_sums, kernels

def output_results( activation_sums, kernels, mtps, compact_scores, cov_scores, notematrix, plot ):
    idxs = sorted( [idx for idx in range( len( kernels ) )], key=lambda i: activation_sums[i], reverse=True )
    for j, idx in enumerate( idxs ):
        # show the score with pattern notes color coded
        if j < 10:
            pattern = mtps[idx]
            score = notematrix.music_score
            notes = [notematrix.loc2point[pit, off] for pit, off in pattern]
            for note in score.flat.notes:
                note.style.color = None
            pattern_notes = {notes[i] : pattern[i] for i in range(len(pattern))}
            for note in score.flat.notes:
                if note in pattern_notes:
                    note.style.color = 'blue'
            t = ''
            # for n in sorted(pattern_notes.keys(), key=lambda x : x.offset):
            #     t += '{}, {}\n'.format( n.pitch, n.offset )
            # score.metadata = music21.metadata.Metadata( movementName=t)
            score.show()

        # print the pattern's info
        if j < 100:
            print 'siatec rank: {}\tsiatec score: {}\tactivation score: {}\npoints: {}\ntrans. vecs: {}'.format(
                idx, cov_scores[idx] * compact_scores[idx], activation_sums[idx],
                sorted( [(notematrix.loc2point[pitch, offset], pitch, notematrix.ts2ql( offset ))
                         for pitch, offset in mtps[idx]],
                        key=lambda x: x[-1] ),
                [(pitch, notematrix.ts2ql( offset )) for pitch, offset in notematrix.mtp_table[mtps[idx]]]
            )
            pad = (int( floor( kernels[idx].shape[0] / 2 ) ), int( floor( kernels[idx].shape[1] / 2 ) ))
            out = convolve_single_channel( torch.from_numpy( notematrix.mat ), torch.from_numpy( kernels[idx] ), pad )
            if plot:
                plot_mat( out, notematrix )

def main(args):
    filename = args[1]
    plot = False
    if len(args) > 2:
        if args[2] == '-p':
            plot = True

    score = music21.converter.parse(filename)
    notematrix = NoteMatrix(score)
    print 'sorting MTPs...'
    results = notematrix.get_sorted_mtps()
    mtps, comp, cov = zip(*results)
    activation_sums, kernels = do_convolutions( mtps[:10], notematrix )
    output_results( activation_sums, kernels, mtps, comp, cov, notematrix, plot )

if __name__ == '__main__':
    main(sys.argv)