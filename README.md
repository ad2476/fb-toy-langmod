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

### Building a corpus for HMM-based generation:

The Hidden Markov Model trains on labeled text, so you will have to first create a POS-tagged corpus. This is relatively easily done as follows:

1. Using `build_corpus.py`, construct your unlabeled corpus as above
2. Run `python` at the interpreter:
    ```
    >>> import build_corpus
    >>> with open("out.txt","r") as f:
    >>>   text = [l for l in f]
    >>> corpus = build_corpus.cleanTextCorpus(text)
    >>> build_corpus.tagCorpus(corpus,"data/<name>.txt")
    ```
3. Your POS-tagged corpus will be where you saved it

## Running the generator:

Run the bigram-based generator: `name.py [-h] [-m MODEL] [-s SAVE] corpus`

Run the HMM-based generator: `usage: hmm_name.py [-h] corpus`. This unfortunately cannot save/load its model as lambdas are not pickle-able.

## (Actual) sample output:

```
arun> hermit
arun> squashingly
arun> AYO CHICKEN ON BOARD THE BUNNIES
arun> Less like linguistics project than how ai hw this begs the landing
arun> Aragorn was leaning communist
```

Profound indeed.
