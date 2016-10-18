# fb-toy-langmod
Just a toy langmod to train off downloaded fb message data. As a small hobby project, expect this to break. Often. Actually, expect nothing and then you can't be disappointed!

## Requirements:

Requires:
* lxml for Python 3
* nltk for Python 3
* Numpy for Python 2 and/or 3
* Tensorflow for Python 3 (Optional)

Tensorflow is required if you're using the feedforward neural network in `nn_name.py` as your model.

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
```python
>>> import build_corpus
>>> with open("out.txt","r") as f:
>>>   text = [l for l in f]
>>> corpus = build_corpus.cleanTextCorpus(text)
>>> build_corpus.tagCorpus(corpus,"data/<name>.txt")
```
3. Your POS-tagged corpus will be where you saved it

## Running the generator:

Run the Markov-based generator: `usage: bgm_name.py [-h] [-m MODEL] [-s SAVE] corpus`

Run the HMM-based generator: `usage: hmm_name.py [-h] corpus`. This unfortunately cannot save/load its model as lambdas are not pickle-able.

Run the Feedforward NN-based generator: `usage: nn_name.py [-h] [-l LOAD | -s SAVE] corpus`. I'd highly recommend saving the model as training can take a while. It's not worth waiting 30 minutes every time in order read some gibberish that approximates your chats.

## (Actual) sample output:

```
arun> hermit
arun> squashingly
arun> AYO CHICKEN ON BOARD THE BUNNIES
arun> Less like linguistics project than how ai hw this begs the landing
arun> Aragorn was leaning communist
```

Profound indeed.
