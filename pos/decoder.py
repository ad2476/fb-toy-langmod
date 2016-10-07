# Hidden Markov Model decoders
import sys
from hmm import STOP

class ViterbiDecoder:

  """ Construct the decoder by passing a hidden markov model. """
  def __init__(self, hmm):
    self.hmm = hmm

  """ Decode a given sentence (as a list)
      
      Returns a sequence of part of speech tags for the input sentence
  """
  def decode(self, sentence):
    decoded = []

    n = len(sentence)

    prev_mu = 1.0
    yprime = STOP
    states = self.hmm.getLabels()
    for i in xrange(0,n): # iterate over the sentence
      word = sentence[i]

      argmax = (None, 0.0)
      for y in states: # argmax over every possible state y
        _, maxmu = argmax
        mu = prev_mu*self.hmm.getSigma(yprime,y)*self.hmm.getTau(y,word)

        if mu >= maxmu:
          argmax = (y,mu)

      # Update step:
      maxy, maxmu = argmax

      prev_mu = maxmu # update prev_mu
      yprime = maxy # update yprime
      decoded.append(maxy) # build decoded sequence

    return decoded

