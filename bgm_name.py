#!/usr/bin/env python2

import argparse
import os
import sys
import bigram
import cPickle
import numpy as np
import string

MAX_LENGTH = 10

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
      if np.random.random() < 0.4:
        first = w_
  if first == "": # this hopefully won't infinite loop...
    return pickFirstWord(model, model.STOP)

  return first

def genSentence(model, lastWord=None):
  words = list(model.getWordSet())
  #word = pickFirstWord(model, (lastWord or model.STOP))
  word = model.STOP
  sentence = []
  i = 0
  transitions = np.zeros(len(words))
  while (i<MAX_LENGTH):
    #sentence.append(word)
    best = (float("-inf"), "") # (max theta_{x,x'}, argmax theta_{x,x'})

    for j,x_ in enumerate(words): # iterate over possible next words
      pair = word,x_
      transitions[j] = np.e**model._thetaFunction(pair)

    transitions /= transitions.sum() # normalise
    word = np.random.choice(words, p=transitions) # random sample next word from transitions
    if word == model.STOP: break
    sentence.append(word)
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
    model.train((2.2,100.0)) # train model with these smoothing params

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
