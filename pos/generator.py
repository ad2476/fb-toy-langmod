# Hidden Markov Model decoders
import sys
import numpy as np
from hmm import STOP

class EmissionSequenceGenerator:

  """ words: a set of words seen in the corpus
      hmm: a hidden markov model trained on the corpus
  """
  def __init__(self, words, hmm, langmod):
    self._words = list(words)
    self._hmm = hmm
    self._langmod = langmod

  """ Generates a sequence of emissions from the HMM, which is
       based on a most likely sequence of labels, and the most likely
       emissions from those labels.
  """
  def generateSentence(self, length_cap):
    label_hash = self._hmm.getLabelHash()
    sigma, tau = self._hmm.getDistribution()
    sentence = [] # output sentence
    word = STOP # usually the prev word when important
    y = label_hash[STOP]

    i = 0
    emissions = np.zeros(len(self._words)) # only allocate this space once
    while i<=length_cap:
      sigma_thresh = self._hmm.sigmaSmoothFactorUnk(y)
      sigma_slice = sigma[y, :] # slice for labels following y
      y = np.random.choice(sigma_slice.size, p=sigma_slice) # random sample y from sigma

      for j,w in enumerate(self._words): # maximise tau_{y,w}
        pair = (y,hash(w))
        emissions[j] = tau[pair]

      emissions /= emissions.sum() # normalise
      word = np.random.choice(self._words, p=emissions) # random sample word from tau, given y

      if word == STOP:
        break
      sentence.append(word)
      i += 1

    return sentence[:-1]

  """ Generates gaussian noise """
  def _noise(self, stddev, mu):
    return stddev*np.random.randn() + mu

  """ Selects a label following STOP at random. This is repetitive w.r.t.
        generateSentence() but fuck it
  """
  def _selectFirstLabel(self):
    sigma,_ = self._hmm.getDistribution() # numpy matrix, uses hashed labels as indices

    # this is the smoothed sigma for an unseen pair, use it
    # to know if the current y' actually does follow STOP:
    sigmaNotStart = self._hmm.sigmaSmoothFactorUnk(STOP)

    label_hash = self._hmm.getLabelHash() # maps string label to int label
    stop_slice = sigma[label_hash[STOP],:] # slice of sigma for STOP,y' transitions
    candidates = [] # list of candidate labels
    for label,y in label_hash.iteritems(): # iterate over hashed label values
      if stop_slice[y] > sigmaNotStart+self._noise(0.1,0.09): # if this label actually comes after STOP
        candidates.append(label)
    
    return np.random.choice(candidates) # select a candidate at random

