# -*- coding: utf-8 -*-

from collections import defaultdict
import itertools
import numpy as np
import sys

from . import _common as common
from . import unk

""" A Hidden Markov Model constructed from visible data """
class VisibleDataHMM:

  """ Construct the HMM object using the outputs and labels (and wordcounts)
        unker: A subclass inheriting from AbstractUnker
        tags: A list of tag sequences (should correspond to the structure of
              the word corpus i.e. a list of sentences-as-lists)
        wordCount: count of unique words in the corpus
  """
  def __init__(self, unker, tags, wordCount):
    # hash x for compatibility with HiddenDataHMM:
    self._outputs = [[hash(x) for x in sentence] for sentence in unker.getUnkedCorpus()]
    self._unker = unker # keep the unker so we can also count the original words
    self._labels = tags # list of list of states
    self._wc = wordCount

    self._sigma = None # not yet defined - don't know how many states there are
    self._tau = None # also not yet defined, need n_ycirc
    self.n_sentences = len(tags)
    if self.n_sentences != len(self._outputs): # problem
      raise ValueError("Outputs and labels should be the same size")

  """ Train the HMM by building the sigma and tau mappings.
        - params is an alpha value to use (default is 1.0)
  """
  def train(self, params=1.0):
    self._alpha = params # add-alpha smoothing

    n_yy_ = defaultdict(int) # n_y,y' (number of times y' follows y)
    n_ycirc = defaultdict(int) # n_y,o (number of times any label follows y)
    n_yx = defaultdict(int) # n_y,x (number of times label y labels output x)

    # build counts
    for i in xrange(len(self._outputs)): # iterate over each sentence
      words = self._outputs[i]
      tags = self._labels[i]
      n = len(words)
      for j in xrange(0, n - 1): # iterate over each word/tag in the sequence
        y = tags[j]
        y_ = tags[j+1] # y_ = y'
        ytuple = (y,y_) # label transition

        x = words[j] # corresponding output
        orig = hash(self._unker.getOrigWord(i,j)) # hash this!
        if x != orig: # this word was UNKed
          yorig = (y,orig)
          n_yx[yorig] += 1 # keep track of the original word, too
          
        yxtuple = (y,x) # emission

        # increment the number of times we've seen these:
        n_yy_[ytuple] += 1
        n_ycirc[y] += 1

        n_yx[yxtuple] += 1

    # properties of this HMM:
    self.tagset = n_ycirc.keys()
    self._labelHash = common.makeLabelHash(self.tagset)
    self.tagsetSize = len(self.tagset)
    self._n_yy_ = n_yy_
    self._n_ycirc = n_ycirc

    # compute sigma matrix:
    self._sigma = np.zeros([self.tagsetSize]*2)
    for y in self.tagset: # first, set up smoothing:
      yhash = self._labelHash[y]
      self._sigma[yhash,:] = self.sigmaSmoothFactorUnk(y) # smooth sigma if the pair (y,y') dne
    for pair,count in n_yy_.iteritems(): # next, initialise sigmas for known (y,y') pairs
      y,yprime = pair
      yhash, yprimehash = self._labelHash[y],self._labelHash[yprime]
      self._sigma[yhash,yprimehash] = (count+self._alpha)/(n_ycirc[y]+self._alpha*self.tagsetSize)
    
    # compute tau dict:
    self._n_yx = defaultdict(float)
    self._tau = common.TauDict(self._alpha, n_ycirc, self._wc)
    for pair,count in n_yx.iteritems():
      y,x = pair
      yhash = self._labelHash[y]
      self._tau[(yhash,x)] = (count+self._alpha)/(n_ycirc[y]+self._alpha*self.tagsetSize) # smooth
      self._n_yx[(yhash,x)] = count # class-wide dict should use hashed labels

  """ Return the smoothing factor for the label y where 
       the count of y,y' is zero
  """
  def sigmaSmoothFactorUnk(self, y):
    return self._alpha/(self._n_ycirc[y]+self._alpha*self.tagsetSize)

  """ Return the sigma_{y,y'} for given y and y' - or 0 if dne """
  def getSigma(self, y, yprime):
    yhash = self._labelHash[y]
    yprimehash = self._labelHash[yprime]
    return self._sigma[yhash,yprimehash]

  """ Return the tau_{y,x} for given y and x - or 0 if dne """
  def getTau(self, y, x):
    y = self._labelHash[y]
    x = hash(self._unker.evaluateWord(x)) # check if x should be unked and hash
    return self._tau[(y,x)]

  """ Return a copy of the labels of this HMM """
  def getLabels(self):
    return set(self.tagset)

  """ Return a copy of the internal mapping of str y -> int i """
  def getLabelHash(self):
    return dict(self._labelHash)

  """ Return the trained internal distributions sigma and tau """
  def getDistribution(self):
    return (self._sigma, self._tau)

  """ Return the number of unique words in this HMM's corpus """
  def getWordCount(self):
    return self._wc

  """ Return the transition/emission counts from the visible data.
      Converts n_y,y' and n_y,o to numpy arrays (nxn and 1xn resp.)
  """
  def getVisibleCounts(self):
    n_yy_mat = np.zeros([self.tagsetSize]*2)
    n_ycircmat = np.zeros(self.tagsetSize)

    """ Iterate through n_yy and copy into array matrix """
    for pair,count in self._n_yy_.iteritems():
        y,yprime = pair
        yhash, yprimehash = self._labelHash[y], self._labelHash[yprime]
        n_yy_mat[yhash,yprimehash] = count
        n_ycircmat[yhash] += count

    return (self._n_yx, n_yy_mat, n_ycircmat)
