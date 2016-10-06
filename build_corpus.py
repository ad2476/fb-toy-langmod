import sys
from lxml import html
import nltk

me = "Arun Drelich" # this is me.

if __name__ == "__main__":

  if len(sys.argv) != 2:
    print("python3 build_corpus.py <message_data>")
    sys.exit(1)

  fname = sys.argv[1]
  print("Loading file...")
  with open(fname, "rb") as f:
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

