# -*- coding: utf-8 -*-
import re
import itertools
from hmm import STOP

""" This class lays out the general interface for a preparser and implements some common
    methods.
"""
class AbstractPreparser:

  def __init__(self, inputData):
    raise NotImplementedError('Subclasses must override this method!')

  """ Parse the outputs and tags into separate lists. """
  def parseWordsTags(self):
    words = []
    tags = []

    try:
      for line in self._rawdata:
        line = "%s %s %s" % (self._stopPair, line, self._stopPair)
        words.append(self.getSentenceWords(line))
        tags.append(self.getSentenceTags(line))

    except IndexError:
      return None

    return words,tags

  """ For use when just the words are desired from a corpus. Aka just tokenise the sentences."""
  def parseWords(self):
    words = []
    for line in self._rawdata:
      line = "%s %s %s" % (STOP, line, STOP)
      words.append(line.split())

    return words

  def writeCorpusWithoutTags(self, out):
    raise NotImplementedError('Subclasses must override this method!')
  @staticmethod
  def getSentenceWords(line):
    raise NotImplementedError('Subclasses must override this method!')
  @staticmethod
  def getSentenceTags(line):
    raise NotImplementedError('Subclasses must override this method!')
  @staticmethod
  def formatOutput(words, tags):
    raise NotImplementedError('Subclasses must override this method!')

""" This class will pre-parse POS-tagged files in the format of the WSJ data """
class EnglishWSJParser(AbstractPreparser):

  """ - inputData: List of sentences in the corpus
  """
  def __init__(self, inputData):
    self._rawdata = inputData
    self._stopPair = STOP + " " + STOP

  @staticmethod
  def getSentenceWords(line):
    return line.split()[::2]

  @staticmethod
  def getSentenceTags(line):
    return line.split()[1::2]

  def writeCorpusWithoutTags(self, out):
    f = open(out, 'w')
    for line in self._rawdata:
      words = cls.getSentenceWords(line)
      for word in words:
        f.write("%s "%word)
      f.write("\n")
    f.close()

  @staticmethod
  def formatOutput(words, tags):
    output = ""
    for word,tag in itertools.izip(words, tags):
      output += word + " " + tag + " "

    return output

class SanskritJNUParser(AbstractPreparser):
  
  def __init__(self, inputData):
    self._rawdata = inputData
    self._stopPair = "%s[%s]" % (STOP, STOP)

  @staticmethod
  def getSentenceWords(line):
    # each line is formatted: "WORD[TAG] WORD[TAG] WORD[TAG]DANDA[TAG]\n"
    #
    # capturing group before a literal '[' char:
    #  match at least 0 times,lazy, on a group consisting of:
    #   not whitespace, not a ']' char
    return re.findall(r"([^\s\]]*?)\[", line)

  @staticmethod
  def getSentenceTags(line):
    # each line is formatted: "WORD[TAG] WORD[TAG] WORD[TAG]DANDA[TAG]\n"
    #
    # capturing group before a literal '[' char:
    #  match at least 0 times,lazy, on a group consisting of:
    #   not whitespace, not a ']' char
    return re.findall(r"\[(.*?)\]", line)

  def writeCorpusWithoutTags(self, out):
    f = open(out, 'w')
    for line in self._rawdata:
      words = self.getSentenceWords(line)
      for word in words:
        f.write("%s "%word)
      f.write("\n")
    f.close()

  @staticmethod
  def formatOutput(words, tags):
    n = len(words)
    output = ""
    for i in xrange(0, n):
      output += "%s[%s] " % (words[i], tags[i])

    return output

    
