import cPickle as pickle
import lda
import os
from serial_lda_gibbs import LdaSampler
from multicore_lda_gibbs import MulticoreLdaSampler
from gpu_lda_gibbs import GPULdaSampler
import time
import numpy as np

pickle_filepath = 'baseline_data.pickle'
N_TOPICS = 10
DOCUMENT_LENGTH = 100
MAXITER = 50
P = 2

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


def serial_gibbs(X, k, iters=50, log=True):
    sampler = LdaSampler(k)
    start = time.time()
    for it, phi in enumerate(sampler.run(X, maxiter=iters)):
        if log:
            print "Iteration", it
            print "Likelihood", sampler.loglikelihood()
        else:
            i = it
    end = time.time()
    print 'Completed %d iterations in %.3f seconds (serial)' % (iters, end - start)
    return sampler


def multicore_gibbs(X, k, p, iters=50, log=True):
    sampler = MulticoreLdaSampler(k, p)
    start = time.time()
    for it, phi in enumerate(sampler.run(X, maxiter=iters)):
        if log:
            print "Iteration", it
            print "Likelihood", sampler.loglikelihood()
        else:
            i = it
    end = time.time()
    print 'Completed %d iterations in %.3f seconds (P=%d)' % (iters, end - start, p)
    return sampler

def gpu_gibbs(X, k, p, iters=50):
    sampler = GPULdaSampler(k, p)
    start = time.time()
    for it, phi in enumerate(sampler.run(X, maxiter=iters)):
        print "Iteration", it
        # print "Likelihood", sampler.loglikelihood()
    end = time.time()
    print 'Completed %d iterations in %.3f seconds (P=%d)' % (iters, end - start, p)

def gpu_gibbs(X, k, p, iters=50):
    sampler = GPULdaSampler(k, p)
    start = time.time()
    for it, phi in enumerate(sampler.run(X, maxiter=iters)):
        print "Iteration", it
        print "Likelihood", sampler.loglikelihood()
    end = time.time()
    print 'Completed %d iterations in %.3f seconds (P=%d)' % (iters, end - start, p)


def truncate_data(X, width, height):
    return np.array([X[height][:width] for row in range(height)])

if __name__ == '__main__':
    X, vocab, titles = load_reuters_dataset()
    print X[0:200,0:4000].shape, len(vocab[0:4000]), len(titles[0:200])
    X, vocab, titles = X[0:200,0:4000], vocab[0:4000], titles[0:200]
    # baseline(X, 10)
    # multicore_gibbs(X, 10, 16)
    # X = np.ones((10, 10))
    print X.shape
    X = X.astype(np.int32)
    gpu_gibbs(X, N_TOPICS, 100, iters=MAXITER)


    