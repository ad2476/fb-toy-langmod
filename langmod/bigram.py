import math
import unigram

class BigramLangmod:

  """ Constructor sets some admin stuff """
  def __init__(self, train, heldout):
    self.STOP = "**@sToP@**" # The stop symbol
    self.trainCorpus = train
    self.heldoutCorpus = heldout
    self._trained = False

    self.unigramModel = unigram.UnigramLangmod(train, heldout)

  """ Build bigram counts from a corpus, aka n_{w,w'}(d)
      Inputs:
        - document: An input corpus to build counts for
                    This can be a list of string, or a file object
      Returns:
        - A tuple of (counts, totalWordCount)
  """
  def buildCounts(self, document):
    counts = {} # map (string,string) -> int: aka n_{w,w'}(d) forall w,w'
    innerCounts = {} # This is n_{w,o} mapping for all w: string->int

    # Populate n_{w,w'}(d) aka bigram counts
    for line in document:
      words = line.split()
      words.append(self.STOP) # Append stop condition to the end
      prevWord = self.STOP # Each line starts with a stop condition
      for word in words:
        n = counts.get((prevWord, word), 0) + 1
        counts[(prevWord, word)] = n

        n_wcirc = innerCounts.get(prevWord, 0) + 1
        innerCounts[prevWord] = n_wcirc

        prevWord = word

    if isinstance(document, file):
      document.seek(0) # Return file pointer to beginning

    return counts, innerCounts

  """ Train a smoothed bigram model given n_{w,w'} and a unigram model.
       This builds the mapping \\Theta_{w,w'} of smoothed log probabilities
      Inputs:
        - alpha (optional): Specify an alpha value, or none to use optimal alpha
      Returns: Nothing
  """
  def train(self, trainParams=None):
    self.bigramCounts, self.n_wcirc = self.buildCounts(self.trainCorpus)

    if trainParams is None:
      self.unigramModel.train() # Optimise unigram alpha
      self.beta = self._optimSmoothParams()
    else:
      alpha, self.beta = trainParams # Use given beta
      self.unigramModel.train(alpha) # Unigram uses given alpha

    self._trained = True

  def getWordSet(self):
    return self.unigramModel.n_w.keys()

  def getBigramSet(self):
    return self.bigramCounts.keys()

  """ Compute the log likelihood of a given document using this model.
      The file pointer of document, if a file, is reset to the beginning of the file.
      Inputs:
        - document: Input document
      Returns: Log likelihood of given document, or None if model isn't trained
  """
  def logLikelihood(self, document):
    logProb = None

    if self._trained:
      # First, get the count of words in the input corpus
      docCounts, _ = self.buildCounts(document)

      # Next, we iterate over the words in the input corpus
      # to compute log likelihood
      logProb = -1*self._likelihoodFunction(docCounts, self.beta)

    return logProb

  def getSmoothParams(self):
    return (self.unigramModel.alpha, self.beta)

  def textSmoothParams(self):
    retstr = "alpha: " + str(self.unigramModel.alpha) + "\n"
    retstr += "beta: " + str(self.beta)
    return retstr

  """ Find the best beta that maximises the likelihood of the held-out
       data using golden-section search
      Inputs: None
      Returns: The best beta i.e. argmax_b L_d(theta)
  """
  def _optimSmoothParams(self):
    heldoutCounts, _ = self.buildCounts(self.heldoutCorpus)

    goldenRatio = (math.sqrt(5) - 1)/2

    start, end = (1e-10, 1e10)
    c = end - goldenRatio*(end - start) 
    d = start + goldenRatio*(end - start)

    while abs(c - d) > 1e-6:
      right = self._likelihoodFunction(heldoutCounts, c)
      left = self._likelihoodFunction(heldoutCounts, d)
      if right > left:
        end = d
        d = c
        c = end - goldenRatio*(end - start)
      else:
        start = c
        c = d
        d = start + goldenRatio*(end - start)

    return (start+end)/2

  """ The function tilde{Theta}_{w,w'}. Assumes the internal unigram model has been
      trained already.
      Inputs:
        - bigram: The w,w' pair
        - beta: The smoothing factor
      Returns:
        - The smoothed probability of w
  """
  def _thetaFunction(self, bigram, beta=None):
    ugram = self.unigramModel
    w, wprime = bigram
    theta_wprime = math.e**(ugram._thetaFunction(wprime, ugram.alpha)) # theta_{w'}

    _beta = beta or self.beta

    n_bigram = self.bigramCounts.get(bigram, 0)
    n_wcirc = self.n_wcirc.get(w, 0)

    return math.log(n_bigram + _beta*theta_wprime) - math.log(n_wcirc + _beta) 

  """ Computes the positive log prob likelihood given docCounts and alpha """
  def _likelihoodFunction(self, docCounts, beta):
    logProb = 0.0
    # Iterate over all bigrams w,w' in docCounts
    for bigram,count in docCounts.iteritems():
      logProb += count*self._thetaFunction(bigram, beta)

    return logProb

