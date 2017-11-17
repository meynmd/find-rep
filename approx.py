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

def plot_mat(tensor):
    plt.figure()
    # imshow needs a numpy array with the channel dimension
    # as the the last dimension so we have to transpose things.
    plt.imshow(np.flip(tensor.numpy(), 0), aspect='auto')
    plt.show()

def convolve_single_channel(matrix, kernel, pad):
    output = fn.conv2d(Variable(matrix.unsqueeze(0).unsqueeze(0)), Variable(kernel.unsqueeze(0).unsqueeze(0)), padding=pad)
    return output.data.squeeze()

def make_kernel(pattern, amp, pad, sig):
    k = amp * pattern
    k = np.pad(k, pad, 'constant')
    return gaussian_filter(k, sig)

def main(filename):
    score = music21.converter.parse(filename)
    notematrix = NoteMatrix(score)
    pattern_stream = [p for p in score.recurse().getElementsByClass('Part')][1].measures(0,1)
    kernel = make_kernel(NoteMatrix(pattern_stream, notematrix.timestep).mat, 1., 5, 1.25)
    pad = (int(floor(kernel.shape[0] / 2)), int(floor(kernel.shape[1] / 2)))
    out = convolve_single_channel(torch.from_numpy(notematrix.mat), torch.from_numpy(kernel), pad)
    plot_mat(out)

if __name__ == '__main__':
    main(sys.argv[1])