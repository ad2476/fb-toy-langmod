import sys
from lxml import html
import nltk
import argparse

me = "Arun Drelich" # this is me.

def parseProgramArgs():
  parser = argparse.ArgumentParser(description="Build a training corpus from HTML message data")
  parser.add_argument("message_data", help="Path to messages.htm")
  parser.add_argument("-t", "--tag", help="Also tag data and build tagged corpus")

  return parser.parse_args()

""" Takes a raw corpus (list of sentences) and
     cleans it up: strips extra whitespace, tokenises,
     removes non-wordy words
"""
def cleanTextCorpus(corpus):
  text = [nltk.word_tokenize(s.strip().lower()) for s in corpus]
  res = []
  for l in text:
    res.append(list(filter(lambda x: "/" not in x and "=" not in x, l)))

  return res

""" Takes a corpus (list of tokenised sentences)
      and an output file name to write to
"""
def tagCorpus(corpus, outfile):
  tagged = [nltk.pos_tag(s) for s in corpus]
  with open(outfile,"w") as f:
    for s in tagged:
      for pair in s:
        f.write("%s %s "%pair)
      f.write("\n")

if __name__ == "__main__":

  args = parseProgramArgs()

  print("Loading file...")
  with open(args.message_data, "rb") as f:
    page = f.read()

  print("Building DOM tree from HTML...")
  tree = html.fromstring(page) # get a DOM tree from the string

  print("Parsing DOM tree...")
  sp_names = tree.xpath('//span[@class="user"]/text()') # get the names on each message
  p_tags = tree.xpath('//p') # get all p tags

  msgs = []
  print("Tokenising corpus...")
  for i,name in enumerate(sp_names):
    if name == me:
      msgs.append(nltk.word_tokenize(p_tags[i].text or ""))

  print("Writing to file...")
  with open("out.txt", "w") as f:
    for msg in msgs:
      if msg:
        for word in msg:
          f.write(word+" ")
        f.write("\n")

  print("Done.")

