#!/usr/bin/env python2

import argparse
import os
import sys
import bigram
import numpy as np
import string

from pos import hmm, utils, generator
from pos.preparser import EnglishWSJParser

MAX_LENGTH = 10
ALPHA = 1.3

def parseProgramArgs(name):
  parser = argparse.ArgumentParser(description="Your very own hmm-%s!"%name)
  parser.add_argument("corpus", help="Path to training corpus")

  return parser.parse_args()

if __name__ == '__main__':

  name = os.path.normpath(os.path.splitext(sys.argv[0])[0])
  args = parseProgramArgs(name)

  sys.stdout.write("Loading files...")
  sys.stdout.flush()

  train = utils.buildCorpus([args.corpus])
  data = EnglishWSJParser(train).parseWordsTags()
  if data is None:
    sys.stderr.write("Error parsing input: Bad format.\n")
    sys.exit(1)

  sys.stdout.write("\nTraining bigram model...")
  sys.stdout.flush()
  langmod = bigram.BigramLangmod(train, [])
  langmod.train((1.2,30.0))
  print "done"

  sys.stdout.write("Training HMM...")
  sys.stdout.flush()
  words,tags = data
  counts,wc = utils.buildCounts(words)
  model = hmm.VisibleDataHMM(hmm.unk.BasicUnker(words,counts),tags,wc)
  model.train(ALPHA) # train the hmm

  print "ready."
  gen = generator.EmissionSequenceGenerator(counts.keys(),model,langmod)
  while True:
    try:
      sys.stdout.write("%s> ..."%name)
      sys.stdout.flush()
      s = gen.generateSentence(MAX_LENGTH)
      if s:
        res = reduce(lambda x,y: x+" "+y if "'" not in y and y not in string.punctuation else x+y, s)
        #res = "".join([i+" " for i in s]).strip()
        sys.stdout.write("\b\b\b"+res+" ")
        raw_input()
      else:
        sys.stdout.write("\b\b\b\b\b")
        for _ in name:
          sys.stdout.write("\b")
    except EOFError:
      break

