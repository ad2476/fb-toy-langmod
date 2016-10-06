# fb-toy-langmod
Just a toy langmod to train off downloaded fb message data

## Requirements:

Requires:
* lxml for Python 3
* nltk for Python 3
* Numpy for Python 2 and/or 3

Unfortunately it's a bit of a clusterfuck between Python 2 and 3.
Blame poor planning combined with laziness. ¯\\\_(ツ)\_/¯

## Building a corpus:

You will need to download your fb data. This can be done from within Facebook and takes around a day.

Use `build_corpus.py` to construct a corpus from a `messages.htm` file. Then `name.py` can be trained and
run on that.

Rename `name.py` to your name to get your name at the prompt! It's as if it's really you!

## (Actual) sample output:

```
arun> hermit
arun> squashingly
arun> AYO CHICKEN ON BOARD THE BUNNIES
arun> Less like linguistics project than how ai hw this begs the landing
arun> Aragorn was leaning communist
```

Profound indeed.
