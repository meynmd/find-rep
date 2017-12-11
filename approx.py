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
import argparse

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

def do_convolutions( mtps, notematrix, sigma=None, pad=5 ):
    print 'doing convolutions...\n'
    activation_sums, kernels = [], []
    for i, pattern in enumerate(mtps):
        kernel = notematrix.make_kernel( pattern, sigma, 0, 0 )
        # print 'pattern {}, matrix {}, kernel {}'.format(i, notematrix.mat.shape, kernel.shape), '\r'
        # pad = (int( floor( kernel.shape[0] / 2 ) ), int( floor( kernel.shape[1] / 2 ) ))
        pad = (0, 0)
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

def get_approx_occurrences(scorematrix, pattern):
    # kernel = scorematrix.make_kernel(pattern, 0., 5, 5)
    kernel = pattern
    # pad = (int(floor(kernel.shape[0] / 2)), int(floor(kernel.shape[1] / 2)))
    pad = (0, 0)
    out = convolve_single_channel(torch.from_numpy(scorematrix.mat), torch.from_numpy(kernel), pad)
    out = Variable(out.view(1, out.size(0), -1))
    # out = fn.max_pool2d(out, (1, int(1. / scorematrix.timestep)))
    out = out.data.view(out.data.size()[1], -1).numpy()
    pad_d, pad_p = kernel.shape[1] // 2, kernel.shape[0] // 2
    out = np.pad(out, ((pad_p, pad_p), (pad_d, pad_d)), 'constant')
    perfect_score = np.sum(kernel * kernel)
    match_mat = out > 0.5 * perfect_score
    matches = []
    rw, rh = pattern.shape[0] // 2, pattern.shape[1] // 2
    for i in range(match_mat.shape[0]):
        for j in range(match_mat.shape[1]):
            if match_mat[i, j]:
                # corners = [scorematrix.timestep * x for x in [i - rw, j - rh, i + rw, j + rh]]
                # x1, y1, x2, y2 = corners
                matches.append((
                    (scorematrix.abs_pitch(i - rw), scorematrix.ts2ql(j - rh)),
                    (scorematrix.abs_pitch(i + rw), scorematrix.ts2ql(j + rh))
                ))
    return sorted(matches, key=lambda x : x[0][1])

def output_results( activation_sums, kernels, mtps, compact_scores, cov_scores, notematrix, plot ):
    idxs = sorted( [idx for idx in range( len( kernels ) )], key=lambda i: activation_sums[i], reverse=True )
    for j, idx in enumerate( idxs ):
        # show the score with pattern notes color coded
        if j < 10 and plot:
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

def show_occurrences( activation_sums, kernels, mtps, compact_scores, cov_scores, notematrix, plot ):
    idxs = sorted( [idx for idx in range( len( kernels ) )], key=lambda i: activation_sums[i], reverse=True )
    for j, idx in enumerate( idxs ):
        if j < 100:
            matches = get_approx_occurrences(notematrix, kernels[idxs[0]])
            print 'pattern: {}\nvectors: {}\noccurrences: {}\n'.format(
                sorted([(notematrix.loc2point[pitch, offset], pitch, notematrix.ts2ql(offset))
                        for pitch, offset in mtps[idx]],
                        key=lambda x: x[-1]),
                [(pitch, notematrix.ts2ql(offset)) for pitch, offset in notematrix.mtp_table[mtps[idx]]],
                matches
            )
        # show the score with pattern notes color coded
        if j < 10 and plot:
            pattern = mtps[idx]
            score = notematrix.music_score
            notes = [notematrix.loc2point[pit, off] for pit, off in pattern]
            for note in score.flat.notes:
                note.style.color = None
            pattern_notes = {notes[i] : pattern[i] for i in range(len(pattern))}
            for note in score.flat.notes:
                if note in pattern_notes:
                    note.style.color = 'blue'

            out = convolve_single_channel(torch.from_numpy(notematrix.mat), torch.from_numpy(kernels[idx]), (0, 0))
            plot_mat( out, notematrix )

            score.show()


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("scorefile", type=str, help='file containing score')
    parser.add_argument('-p', '--plot', help='plot and show score', action='store_true')
    parser.add_argument('-s', '--sigma', type=float, help='sigma for gaussian blur')
    args = parser.parse_args()
    filename = args.scorefile
    plot = args.plot
    sigma = args.sigma

    score = music21.converter.parse(filename)
    notematrix = NoteMatrix(score, timestep_min=0.25)
    print 'sorting MTPs...'
    results = notematrix.get_sorted_mtps()
    mtps, comp, cov = zip(*results)
    mtps = mtps[:10]
    activation_sums, kernels = do_convolutions( mtps, notematrix, sigma )

    show_occurrences( activation_sums, kernels, mtps, comp, cov, notematrix, plot )

    # idxs = sorted( [idx for idx in range( len( kernels ) )], key=lambda i: activation_sums[i], reverse=True )
    # matches = get_approx_occurrences(notematrix, kernels[idxs[0]])
    #
    # print sorted([(notematrix.loc2point[pitch, offset], pitch, notematrix.ts2ql(offset))
    #         for pitch, offset in mtps[idxs[0]]],
    #         key=lambda x: x[-1])
    # print matches

if __name__ == '__main__':
    main(sys.argv)