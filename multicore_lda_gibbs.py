"""
(C) Mathieu Blondel - 2010
License: BSD 3 clause

Implementation of the collapsed Gibbs sampler for
Latent Dirichlet Allocation, as described in

Finding scientifc topics (Griffiths and Steyvers)
"""
from copy import copy
import numpy as np
from multiprocessing import Pool
import scipy as sp
from scipy.special import gammaln

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
        self.nzw = np.zeros((self.n_topics, vocab_size))
        self.nm = np.zeros(n_docs)
        self.nz = np.zeros(self.n_topics)
        self.topics = {}

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
                self.topics[(m,i)] = z

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



    def _conditional_distribution(self, m, w, p):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.local_nzw[p].shape[1]
        left = (self.local_nzw[p][:,w] + self.beta) / \
               (self.local_nz[p] + self.beta * vocab_size)
        right = (self.nmz[m,:] + self.alpha) / \
                1 #(self.nm[m] + self.alpha * self.n_topics)
        p_z = left * right
        # normalize to obtain probabilities
        p_z /= np.sum(p_z)
        return p_z

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
            
            def sample(p):
                for m in self.docs_by_processor[p]:
                    for i, w in enumerate(word_indices(matrix[m, :])):
                        z = self.topics[(m,i)]
                        self.nmz[m,z] -= 1
                        self.nm[m] -= 1
                        self.local_nzw[p][z,w] -= 1
                        self.local_nz[p][z] -= 1

                        p_z = self._conditional_distribution(m, w, p)
                        if not np.isclose(np.sum(p_z), 1.):
                            print p_z
                        z = sample_index(p_z)

                        self.nmz[m,z] += 1
                        self.nm[m] += 1
                        self.local_nzw[p][z,w] += 1
                        self.local_nz[p][z] += 1
                        self.topics[(m,i)] = z
            
            pool = Pool(processes=self.P)
            for p in range(self.P):
                pool.apply_async(sample, [p])
            pool.close()
            pool.join()

            self._global_update()

            yield self.phi()

if __name__ == "__main__":
    import os
    import shutil

    N_TOPICS = 10
    DOCUMENT_LENGTH = 100
    FOLDER = "topicimg"

    def vertical_topic(width, topic_index, document_length):
        """
        Generate a topic whose words form a vertical bar.
        """
        m = np.zeros((width, width))
        m[:, topic_index] = int(document_length / width)
        return m.flatten()

    def horizontal_topic(width, topic_index, document_length):
        """
        Generate a topic whose words form a horizontal bar.
        """
        m = np.zeros((width, width))
        m[topic_index, :] = int(document_length / width)
        return m.flatten()

    def save_document_image(filename, doc, zoom=2):
        """
        Save document as an image.

        doc must be a square matrix
        """
        height, width = doc.shape
        zoom = np.ones((width*zoom, width*zoom))
        # imsave scales pixels between 0 and 255 automatically
        sp.misc.imsave(filename, np.kron(doc, zoom))

    def gen_word_distribution(n_topics, document_length):
        """
        Generate a word distribution for each of the n_topics.
        """
        width = n_topics / 2
        vocab_size = width ** 2
        m = np.zeros((n_topics, vocab_size))

        for k in range(width):
            m[k,:] = vertical_topic(width, k, document_length)

        for k in range(width):
            m[k+width,:] = horizontal_topic(width, k, document_length)

        m /= m.sum(axis=1)[:, np.newaxis] # turn counts into probabilities

        return m

    def gen_document(word_dist, n_topics, vocab_size, length=DOCUMENT_LENGTH, alpha=0.1):
        """
        Generate a document:
            1) Sample topic proportions from the Dirichlet distribution.
            2) Sample a topic index from the Multinomial with the topic
               proportions from 1).
            3) Sample a word from the Multinomial corresponding to the topic
               index from 2).
            4) Go to 2) if need another word.
        """
        theta = np.random.mtrand.dirichlet([alpha] * n_topics)
        v = np.zeros(vocab_size)
        for n in range(length):
            z = sample_index(theta)
            w = sample_index(word_dist[z,:])
            v[w] += 1
        return v

    def gen_documents(word_dist, n_topics, vocab_size, n=500):
        """
        Generate a document-term matrix.
        """
        m = np.zeros((n, vocab_size))
        for i in xrange(n):
            m[i, :] = gen_document(word_dist, n_topics, vocab_size)
        return m

    if os.path.exists(FOLDER):
        shutil.rmtree(FOLDER)
    os.mkdir(FOLDER)

    width = N_TOPICS / 2
    vocab_size = width ** 2
    word_dist = gen_word_distribution(N_TOPICS, DOCUMENT_LENGTH)
    matrix = gen_documents(word_dist, N_TOPICS, vocab_size)
    sampler = LdaSampler(N_TOPICS)

    for it, phi in enumerate(sampler.run(matrix)):
        print "Iteration", it
        print "Likelihood", sampler.loglikelihood()

        if it % 5 == 0:
            for z in range(N_TOPICS):
                save_document_image("topicimg/topic%d-%d.png" % (it,z),
                                    phi[z,:].reshape(width,-1))

