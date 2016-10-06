#!/usr/bin/env python2

import argparse
import os
import sys
import bigram
import cPickle
import numpy as np
import string

MAX_LENGTH = 20

def parseProgramArgs(name):
  parser = argparse.ArgumentParser(description="Your very own bigram-%s!"%name)
  parser.add_argument("corpus", help="Path to training corpus")
  parser.add_argument("-m", "--model", help="pre-trained model")
  parser.add_argument("-s", "--save", help="Save the model to a file")

  return parser.parse_args()

def pickFirstWord(model, prev):
  first = ""
  bigrams = model.getBigramSet()
  np.random.shuffle(bigrams)
  for pair in bigrams:
    w,w_ = pair
    if w == prev:
      if np.random.random() < 0.7:
        first = w_
  if first == "": # this hopefully won't infinite loop...
    return pickFirstWord(model, model.STOP)

  return first

def genSentence(model, lastWord=None):
  word = pickFirstWord(model, (lastWord or model.STOP))
  sentence = []
  i = 0
  while (i<MAX_LENGTH) and (word!=model.STOP):
    sentence.append(word)
    best = (float("-inf"), "") # (max theta_{x,x'}, argmax theta_{x,x'})
    for pair in model.getBigramSet():
      x,x_ = pair
      if x == word: # only get the words following this one
        theta = model._thetaFunction((word,x_))*(0.5*np.random.randn()+1.0) # put in some noise
        maxtheta, _ = best
        if theta >= maxtheta:
          best = (theta,x_)

    _, word = best
    i+=1

  return sentence

if __name__ == '__main__':

  name = os.path.normpath(os.path.splitext(sys.argv[0])[0])
  args = parseProgramArgs(name)

  sys.stdout.write("Loading files...")
  sys.stdout.flush()
  if not args.model:
    with open(args.corpus, "r") as train_file:
      train = [line for line in train_file]

    sys.stdout.write("\nTraining language model...")
    sys.stdout.flush()
    model = bigram.BigramLangmod(train, [])
    model.train((1.2,200.0)) # train model with these smoothing params

    if args.save:
      with open(args.save, "wb") as output:
        cPickle.dump(model, output) # dump model to file
  else:
    with open(args.model, "rb") as f:
      model = cPickle.load(f)

  print "ready."
  lastWord = model.STOP
  while True:
    try:
      sys.stdout.write("%s> ..."%name)
      sys.stdout.flush()
      s = genSentence(model,lastWord)
      if s:
        #lastWord = s[-1]
        #res = reduce(lambda x,y: x+" "+y if "'" not in y and y not in string.punctuation else x+y, s)
        res = "".join([i+" " for i in s]).strip()
        sys.stdout.write("\b\b\b"+res+" ")
      raw_input()
    except EOFError:
      break
