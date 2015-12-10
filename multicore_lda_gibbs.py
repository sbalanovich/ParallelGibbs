"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""
from copy import copy
import ctypes
from functools import partial
import numpy as np
from multiprocessing import Array, Pool
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

def make_shared_array(shape, _type=ctypes.c_int):

    shared_array_base = Array(_type, np.prod(shape), lock=False)
    shared_array = np.ctypeslib.as_array(shared_array_base)
    shared_array = shared_array.reshape(*shape)

    return Array(_type, shape, lock=False)


n_topics = 10
P = 1
n_docs = 395
vocab_size = 4258

nmz = make_shared_array((n_docs, n_topics))
# number of times topic z and word w co-occur

nzw = make_shared_array((n_topics, vocab_size))
nm = make_shared_array((n_docs,))
nz = make_shared_array((n_topics,))
topics = make_shared_array((n_docs, vocab_size))

local_nzw = make_shared_array((P, n_topics, vocab_size))
local_nz = make_shared_array((P, n_topics))


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
        global nmz, nzw, nm, nz, topics, local_nzw, local_nz
        # translation of AD-LDA paper notation to variable names
        # W - vocab_size
        # nk - nz
        # nk|j = nmz[m,:]
        # nx|k = nzw[:,w]

        n_docs, vocab_size = matrix.shape

        self.docs_by_processor = np.array_split(np.arange(n_docs), self.P)

        
        # number of times document m and topic z co-occur

        for m in xrange(n_docs):
            # i is a number between 0 and doc_length-1
            # w is a number between 0 and vocab_size-1
            for i, w in enumerate(word_indices(matrix[m, :])):
                # choose an arbitrary topic as first topic for word i
                z = np.random.randint(self.n_topics)
                nmz[m,z] += 1
                nm[m] += 1
                nzw[z,w] += 1
                nz[z] += 1
                topics[m,i] = z

                for p in range(self.P):
                    local_nzw[p, z, w] += 1
                    local_nz[p, z] += 1

    def _global_update(self):
        """Reduce scatter operation to update"""

        global nmz, nzw, nm, nz, topics, local_nzw, local_nz

        vocab_size = nzw.shape[1]
        for w in xrange(vocab_size):
            update = sum([local_nzw[p][:,w] - nzw[:, w] for p in range(self.P)])
            nzw[:, w] = nzw[:, w] + update

        for z in xrange(self.n_topics):
            update = sum([local_nz[p][z] - nz[z] for p in range(self.P)])
            nz[z] = nz[z] + update

        # local_nzw = [copy(nzw) for p in range(self.P)]
        # local_nz = [copy(nz) for p in range(self.P)]

        for w in xrange(vocab_size):
            for p in range(self.P):
                local_nzw[p][:, w] = nzw[:, w]

        for z in xrange(self.n_topics):
            for p in range(self.P):
                local_nz[p][z] = nz[z]

    def loglikelihood(self):
        """
        Compute the likelihood that the model generated the data.
        """
        vocab_size = nzw.shape[1]
        n_docs = nmz.shape[0]
        lik = 0

        for z in xrange(self.n_topics):
            lik += log_multi_beta(nzw[z,:]+self.beta)
            lik -= log_multi_beta(self.beta, vocab_size)

        for m in xrange(n_docs):
            lik += log_multi_beta(nmz[m,:]+self.alpha)
            lik -= log_multi_beta(self.alpha, self.n_topics)

        return lik

    def phi(self):
        """
        Compute phi = p(w|z).
        """
        V = nzw.shape[1]
        num = nzw + self.beta
        num /= np.sum(num, axis=1)[:, np.newaxis]
        return num

    def run(self, matrix, maxiter=30):
        """
        Run the Gibbs sampler.
        """
        global nmz, nzw, nm, nz, topics, local_nzw, local_nz
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)

        for it in xrange(maxiter):

            start = time.time()
            results = []
            for p in range(self.P):
                args = [matrix[self.docs_by_processor[p]], self.docs_by_processor,
                self.alpha, self.beta, p]
                results.append(self.pool.apply_async(sample, args))

            for p, res in enumerate(results):
                done = res.get()
            #     indices = self.docs_by_processor[p]
            #     local_nz[p], local_nzw[p] = nz, nzw
            #     nm[indices] = nm
            #     nmz[indices] = nmz
            #     topics[indices] = topics

            end = time.time()
            self.sample_times.append(end-start)
            print 'Sampled in %.3f seconds' % (end - start)

            start = time.time()
            self._global_update()
            end = time.time()
            self.update_times.append(end-start)
            print 'Updated in %.3f seconds' % (end - start)

            yield 1#self.phi()

def conditional_distribution(m, w, p, alpha, beta):
    """
    Conditional distribution (vector of size n_topics).
    """
    global nmz, nzw, nm, nz, topics, local_nzw, local_nz
    vocab_size = local_nzw[p].shape[1]
    left = (local_nzw[p][:,w] + beta) / \
           (local_nz[p] + beta * vocab_size)
    right = (nmz[m,:] + alpha) / \
            1 #(nm[m] + self.alpha * self.n_topics)
    p_z = left * right
    # normalize to obtain probabilities
    p_z /= np.sum(p_z)
    return p_z


def sample(matrix_slice, docs_by_processor, alpha, beta, p):
    global nmz, nzw, nm, nz, topics, local_nzw, local_nz
    for word_number, m in enumerate(docs_by_processor[p]):
        for i, w in enumerate(word_indices(matrix_slice[word_number])):
            z = topics[m,i]
            nmz[m,z] -= 1
            nm[m] -= 1
            local_nzw[p][z,w] -= 1
            local_nz[p][z] -= 1

            p_z = conditional_distribution(m, w, p, alpha, beta)
            if not np.isclose(np.sum(p_z), 1.):
                print p_z
            z = sample_index(p_z)

            nmz[m,z] += 1
            nm[m] += 1
            local_nzw[p][z,w] += 1
            local_nz[p][z] += 1
            topics[m,i] = z
    return 1
