import click
import cPickle as pickle
import lda
import os
from serial_lda_gibbs import LdaSampler
from multicore_lda_gibbs import MulticoreLdaSampler
from gpu_lda_gibbs import GPULdaSampler
import time
import numpy as np

pickle_filepath = 'baseline_data.pickle'
N_TOPICS = 5
DOCUMENT_LENGTH = 100
MAXITER = 50
P = 2


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


def load_from_file(dataset):
    files = {
        'nips': 'docword.nips.txt', 
        'nytimes': 'docwords.nytimes.txt',
    }
    if dataset == 'reuters':
        return load_reuters_dataset()[0]
    elif dataset in files: 
        with open(files[dataset], 'r') as rfile:
            lines = rfile.readlines()
        n_documents = int(lines[0])
        n_words = int(lines[1])
        X = np.zeros((n_documents, n_words))
        data = map(lambda s: map(int, s.split()), lines[3:])
        for doc, word, count in data:
            X[doc-1][word-1] = count
        return X
    else:
        raise Exception ('Dataset %s not found' % dataset)

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


def multicore_gibbs(X, k, iters, p, log=True):
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


def gpu_gibbs(X, k, iters, p):
    sampler = GPULdaSampler(k, p)
    start = time.time()
    for it, phi in enumerate(sampler.run(X, maxiter=iters)):
        print "Iteration", it
        print "Likelihood", sampler.loglikelihood()
    end = time.time()
    print 'Completed %d iterations in %.3f seconds (P=%d)' % (iters, end - start, p)


datasets = {
    'synthetic': lambda : np.ones((200, 400)).astype(np.int32),
    'reuters': lambda: load_reuters_dataset()[0],
    'nytimes': lambda: load_from_file('nytimes'),
}

methods = {
    'serial': serial_gibbs,
    'multiprocessing': multicore_gibbs,
    'gpu': gpu_gibbs,
}

data_options = ', '.join(datasets.keys())
method_options = ', '.join(methods.keys())

@click.command()
@click.option('--dataset', default='reuters', prompt='Select a dataset (options include %s). Default is' % data_options, 
    help='Dataset to run sampler on. Options include %s.' % data_options)
@click.option('--method', default='multiprocessing', prompt='Select a method (options include %s). Default is' % method_options,
    help='Sampler implementation to be run. Options include %s.' % method_options)
@click.option('--n_topics', default=10, help='Number of topics (k in the literature)')
@click.option('--P', default=8, prompt='Number of processes to run (ignored for serial). Default is', 
    help='Number of processes (ignored for serial version)')
@click.option('--iterations', default=10, prompt='Number of iterations to run sampler for. Defaut is',
    help='Number of iterations to run sampler for')
def main(dataset, method, n_topics, p, iterations):
    print 'Running the %s sampler on %s for %d iterations...' % (method, dataset, iterations)
    X = datasets[dataset]()
    sampler = methods[method]
    args = (X, n_topics, iterations, )
    if method != 'serial':
        args += (P,)
    sampler(*args)

def test():
    X = (np.random.rand(40, 100) * 10).astype(np.int32)
    # X, vocab, titles = load_reuters_dataset()
    # X, vocab, titles = X[0:10,0:800], vocab[0:4000], titles[0:200]

    sampler = gpu_gibbs
    args = (X, 10, 100, 10)
    sampler(*args)

if __name__ == '__main__':
    test()
    # main('reuters', 'multiprocessing', 10, 8, 2)
    # X, vocab, titles = load_reuters_dataset()
    # print X[0:200,0:4000].shape, len(vocab[0:4000]), len(titles[0:200])
    # X, vocab, titles = X[0:200,0:4000], vocab[0:4000], titles[0:200]
    # baseline(X, 10)
    # multicore_gibbs(X, 10, 16)
    # X = np.ones((200, 400))
    # X = X.astype(np.int32)
    # gpu_gibbs(X, N_TOPICS, 100, iters=MAXITER)


    