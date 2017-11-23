from __future__ import division
from math import *
from copy import deepcopy
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import torch
import torch.nn.functional as fn
from torch.autograd import Variable
import music21
from note_mat import *

def plot_mat(tensor, notematrix):
    # plt.figure()
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

def main(filename):
    score = music21.converter.parse(filename)
    notematrix = NoteMatrix(score)
    pattern_stream = [p for p in score.recurse().getElementsByClass('Part')][1].getElementsByOffset(0, 6) #measures(0, 1)
    kernel = make_kernel(NoteMatrix(pattern_stream, notematrix.timestep).mat, 1., 5, 1.)
    pattern_mat = NoteMatrix( pattern_stream, notematrix.timestep ).mat
    # kernel = np.random.rand(pattern_mat.shape[0], pattern_mat.shape[1])
    pad = (int(floor(kernel.shape[0] / 2)), int(floor(kernel.shape[1] / 2)))
    out = convolve_single_channel(torch.from_numpy(notematrix.mat), torch.from_numpy(kernel), pad)
    plot_mat(out, notematrix)
    out = out.view( 1, out.size( 0 ), -1 )
    out = fn.max_pool2d(out, (1, int(1. / notematrix.timestep)))
    # plot_mat(out.view(out.size(1), -1), notematrix)

    total = np.sum(out.numpy())

if __name__ == '__main__':
    main(sys.argv[1])