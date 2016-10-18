#!/usr/bin/env python3

import os
import argparse
import sys
import pickle
import tensorflow as tf
import numpy as np
import codecs

batch_size = 20
embed_size = 30
hidden_size = 100
MAX_LENGTH = 6

STOP = "*S*"

def parseProgramArgs(name):
  parser = argparse.ArgumentParser(description="Your very own nn-%s!"%name)
  parser.add_argument("corpus", help="Path to training corpus")
  g1 = parser.add_mutually_exclusive_group()
  g1.add_argument("-l", "--load", help="pre-trained tensorflow model")
  g1.add_argument("-s", "--save", help="Save the model to a file")

  return parser.parse_args()

# text is a list of sentences
def makeWordIDs(corpus):
  wordIDs = {}
  index = 0
  for word in corpus:
    if word not in wordIDs:
      wordIDs[word] = index
      index += 1

  return wordIDs # maps word -> int

def genSentence(sess, inpt, probs):
  stopcode = wordIDs[STOP]
  #sentence = [ np.random.randint(vocab_size) ]
  sentence = [ stopcode ]
  maxlen = int(np.random.randn()*2 + MAX_LENGTH)
  while len(sentence) < maxlen:
    word = sentence[-1]
    dist = np.array(probs.eval(feed_dict={inpt: [word]}, session=sess)[0])
    dist /= dist.sum()
    nword = np.random.choice(len(dist),p=dist)

    if nword == stopcode:
      break
    sentence.append(nword)

  return sentence[1:] # exclude stop symbol

if __name__ == '__main__':
  name = os.path.normpath(os.path.splitext(sys.argv[0])[0])
  args = parseProgramArgs(name)

  sess = tf.Session()

  if args.load: # if we're loading from a saved model
    print("Loading model from file...")
    with open("%s.dict"%args.load, "rb") as f:
      wordIDs = pickle.load(f)
      
    vocab_size = len(wordIDs.keys())
    saver = tf.train.import_meta_graph("%s.meta"%args.load)
    saver.restore(sess, args.load) # restore the session

    inpt = tf.get_collection('inpt')[0]
    logits = tf.get_collection('logits')[0]
  else: # otherwise we have to train a model
    print("Loading text corpus...")
    with codecs.open(args.corpus, "r",encoding='utf-8', errors='ignore') as train:
      corpus = [ STOP ]
      for line in train:
        corpus.extend(line.split())
        corpus.append(STOP)

    wordIDs = makeWordIDs(corpus)
    vocab_size = len(wordIDs.keys())
    print(vocab_size)

    corpus = [wordIDs[w] for w in corpus] # convert to ints

    # setup tf neural network (no idea how to modularise tensorflow...):
    inpt = tf.placeholder(tf.int32, [None], name='inpt') # corpus
    output = tf.placeholder(tf.int32,[None], name='output')

    E = tf.Variable(tf.truncated_normal([vocab_size, embed_size], stddev=0.1))
    W1 = tf.Variable(tf.truncated_normal([embed_size, hidden_size], stddev=0.1)) # W_1
    b1 = tf.Variable(tf.constant(0.01,shape=[hidden_size])) # first bias

    W2 = tf.Variable(tf.truncated_normal([hidden_size,vocab_size], stddev=0.1))
    b2 = tf.Variable(tf.constant(0.01,shape=[vocab_size]))

    Elookup = tf.nn.embedding_lookup(E, inpt) 
    relu_layer = tf.nn.relu(tf.matmul(Elookup, W1) + b1)

    logits = tf.matmul(relu_layer, W2) + b2
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, output)

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    perplexity = tf.exp(tf.reduce_mean(cross_entropy))

    sess.run(tf.initialize_all_variables())

    # train the model
    total = int(len(corpus)/batch_size)
    n = 0
    for i in range(0,len(corpus)-batch_size-1,batch_size):
      words = corpus[i:i+batch_size]
      nextwords = corpus[i+1:i+batch_size+1]

      train_step.run(feed_dict={inpt: words, output: nextwords}, session=sess)
      if not n%100:
        print("Batch #%d of %d (%.2f%%):"%(n,total,100*n/total))
        print("\tTrain perplexity: %.6f"%perplexity.eval(feed_dict={inpt:words, output:nextwords}, session=sess))

      n+=1

    if args.save: # if we're saving this model
      tf.add_to_collection('logits', logits) # save logits
      tf.add_to_collection('inpt', inpt) # save input placeholder
      tf.add_to_collection('output', output)
      tf.add_to_collection('perplexity', perplexity) # save perplexity

      saver = tf.train.Saver()

      print("Saving model to disk...")
      p = saver.save(sess, args.save)
      with open("%s.dict"%args.save, "wb") as f:
        pickle.dump(wordIDs, f)
      print("Wrote model to %s"%p)

  noise = tf.random_normal(tf.shape(logits),stddev=1.0)
  #nxt = tf.argmax(logits + noise,1)
  sm = tf.nn.softmax(logits + noise) # convert to probabilities
  inv_map = {v: k for k, v in wordIDs.items()}
  while True:
    try:
      s = [inv_map[w] for w in genSentence(sess, inpt, sm)]
      if s:
        sys.stdout.write("%s> "%name)
        sys.stdout.flush()
        #res = functools.reduce(lambda x,y: x+" "+y if "'" not in y and y not in string.punctuation else x+y, s)
        res = "".join([w+" " for w in s]).strip()
        sys.stdout.write(res)
        input()
    except EOFError:
      break
