"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""
from copy import copy
from functools import partial
import numpy as np
from multiprocessing import Pool
import scipy as sp
from scipy.special import gammaln
import time

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    if not sum(p[:-1]) <= 1:
        print p
    return np.random.multinomial(1,p).argmax()

def word_indices(vec):
    """
    Turn a document vector of size vocab_size to a sequence
    of word indices. The word indices are between 0 and
    vocab_size-1. The sequence length is equal to the document length.
    """
    for idx in vec.nonzero()[0]:
        for i in xrange(int(vec[idx])):
            yield idx

def log_multi_beta(alpha, K=None):
    """
    Logarithm of the multinomial beta function.
    """
    if K is None:
        # alpha is assumed to be a vector
        return np.sum(gammaln(alpha)) - gammaln(np.sum(alpha))
    else:
        # alpha is assumed to be a scalar
        return K * gammaln(alpha) - gammaln(K*alpha)


class MulticoreLdaSampler(object):

    def __init__(self, n_topics, P, alpha=0.1, beta=0.1):
        """
        n_topics: desired number of topics
        alpha: a scalar (FIXME: accept vector of size n_topics)
        beta: a scalar (FIME: accept vector of size vocab_size)
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.P = P
        self.pool = Pool(processes=P)

        self.sample_times = []
        self.update_times = []



    def _initialize(self, matrix):

        # translation of AD-LDA paper notation to variable names
        # W - vocab_size
        # nk - nz
        # nk|j = nmz[m,:]
        # nx|k = nzw[:,w]

        n_docs, vocab_size = matrix.shape

        self.docs_by_processor = np.array_split(np.arange(n_docs), self.P)

        # number of times document m and topic z co-occur
        self.nmz = np.zeros((n_docs, self.n_topics))
        # number of times topic z and word w co-occur
        self.nzw = np.zeros((self.n_topics, vocab_size)).astype(np.int32)
        self.nm = np.zeros(n_docs).astype(np.int32)
        self.nz = np.zeros(self.n_topics).astype(np.int32)
        self.topics = np.zeros((n_docs, vocab_size)).astype(np.int32)

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                self.nmz[m,z] += 1
                self.nm[m] += 1
                self.nzw[z,w] += 1
                self.nz[z] += 1
                self.topics[m,i] = z

        self.local_nzw = [copy(self.nzw) for p in range(self.P)]
        self.local_nz = [copy(self.nz) for p in range(self.P)]

    def _global_update(self):

        vocab_size = self.nzw.shape[1]
        for w in xrange(vocab_size):
            update = sum([self.local_nzw[p][:,w]  - self.nzw[:,w] for p in range(self.P)])
            self.nzw[:,w] = self.nzw[:,w] + update

        for z in xrange(self.n_topics):
            update = sum([self.local_nz[p][z] - self.nz[z] for p in range(self.P)])
            self.nz[z] = self.nz[z] + update

        self.local_nzw = [copy(self.nzw) for p in range(self.P)]
        self.local_nz = [copy(self.nz) for p in range(self.P)]


    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = self.nzw.shape[1]
        n_docs = self.nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(self.nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(self.nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = self.nzw.shape[1]
        num = self.nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def run(self, matrix, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)

        for it in xrange(maxiter):

            start = time.time()
            results = []
            for p in range(self.P):
                args = [self.topics, matrix[self.docs_by_processor[p]], self.docs_by_processor,
                    self.nmz, self.nm, self.local_nzw[p], self.local_nz[p], self.alpha, self.beta, p]
                results.append(self.pool.apply_async(sample, args))

        
            for p, res in enumerate(results):
                nz, nzw, nm, nmz, topics = res.get()
                indices = self.docs_by_processor[p]
                self.local_nz[p], self.local_nzw[p] = nz, nzw
                self.nm[indices] = nm
                self.nmz[indices] = nmz
                self.topics[indices] = topics

            end = time.time()
            self.sample_times.append(end-start)
            print 'Sampled in %.3f seconds' % (end - start)

            start = time.time()
            self._global_update()
            end = time.time()
            self.update_times.append(end-start)
            print 'Updated in %.3f seconds' % (end - start)

            yield 1#self.phi()

def conditional_distribution(m, w, p, nmz, local_nzw, local_nz, alpha, beta):
    """
    Conditional distribution (vector of size n_topics).
    """
    vocab_size = local_nzw.shape[1]
    left = (local_nzw[:,w] + beta) / \
           (local_nz + beta * vocab_size)
    right = (nmz[m,:] + alpha) / \
            1 #(self.nm[m] + self.alpha * self.n_topics)
    p_z = left * right
    # normalize to obtain probabilities
    p_z /= np.sum(p_z)
    return p_z

def sample(topics, matrix_slice, docs_by_processor, nmz, nm, local_nzw, local_nz, alpha, beta, p):
    for word_number, m in enumerate(docs_by_processor[p]):
        for i, w in enumerate(word_indices(matrix_slice[word_number])):
            z = topics[m,i]
            nmz[m,z] -= 1
            nm[m] -= 1
            local_nzw[z,w] -= 1
            local_nz[z] -= 1

            p_z = conditional_distribution(m, w, p, nmz, local_nzw, local_nz, alpha, beta)
            if not np.isclose(np.sum(p_z), 1.):
                print p_z
            z = sample_index(p_z)

            nmz[m,z] += 1
            nm[m] += 1
            local_nzw[z,w] += 1
            local_nz[z] += 1
            topics[m,i] = z
    
    return local_nz, local_nzw, nm[docs_by_processor[p]], nmz[docs_by_processor[p]], topics[docs_by_processor[p]]

