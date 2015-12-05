import cPickle as pickle
import lda
import os
from serial_lda_gibbs import LdaSampler
import time

pickle_filepath = 'baseline_data.pickle'

def main():
    pass


def load_reuters_dataset():
    if not os.path.exists(pickle_filepath):
        dump_reuters_dataset()
    with open(pickle_filepath, 'r') as rfile:
        X, vocab, titles = pickle.load(rfile)
    return X, vocab, titles

def dump_reuters_dataset():
    X = lda.datasets.load_reuters()
    vocab = lda.datasets.load_reuters_vocab()
    titles = lda.datasets.load_reuters_titles()

    with open(pickle_filepath, 'w') as wfile:
        pickle.dump( (X, vocab, titles), wfile)


def baseline(X, k):
    sampler = LdaSampler(k)
    with Timer() as t:
        for it, phi in enumerate(sampler.run(X, maxiter=50)):
            print "Iteration", it
            print "Likelihood", sampler.loglikelihood()

    print 'Fit in %.3f seconds' % t.interval


#  time class from http://preshing.com/20110924/timing-your-code-using-pythons-with-statement/
#  and lots of other places on the web
class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

if __name__ == '__main__':
    X, vocab, titles = load_reuters_dataset()
    baseline(X, 10)
    