import numpy as np
import scipy as sp
from scipy.special import gammaln
import pyopencl as cl
import time

def sample_index(p):
    """
    Sample from the Multinomial distribution and return the sample index.
    """
    return np.random.multinomial(1,p).argmax()

def mod_add(p, l, P):
    return (p + l) % P

def round_up(global_size, group_size):
    r = global_size % group_size
    if r == 0:
        return global_size
    return global_size + group_size - r

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

class Timer:    
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

class GpuLdaSampler(object):
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

        for i, w in enumerate(word_indices(matrix[m, :])):
            update = sum([self.nzw[:,w] - self.local_nzw[p][:,w] for p in range(self.P)])
            self.nzw[:,w] = self.nzw[:,w] + update

            for z in range(n_topics):
                self.nz = sum([self.local_nz[p][z] for p in range(P)])



        self.local_nzw = [copy(self.nzw) for p in range(self.P)]
        self.local_nz = [copy(self.nz) for p in range(self.P)]



    def _conditional_distribution(self, m, w):
        """
        Conditional distribution (vector of size n_topics).
        """
        vocab_size = self.nzw.shape[1]
        left = (self.nzw[:,w] + self.beta) / \
               (self.nz + self.beta * vocab_size)
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
            for m in xrange(n_docs):
                for i, w in enumerate(word_indices(matrix[m, :])):
                    z = self.topics[(m,i)]
                    self.nmz[m,z] -= 1
                    self.nm[m] -= 1
                    self.nzw[z,w] -= 1
                    self.nz[z] -= 1

                    p_z = self._conditional_distribution(m, w)
                    z = sample_index(p_z)

                    self.nmz[m,z] += 1
                    self.nm[m] += 1
                    self.nzw[z,w] += 1
                    self.nz[z] += 1
                    self.topics[(m,i)] = z

            yield self.phi()

if __name__ == '__main__':
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    program = cl.Program(context, open('gibbs.cl').read()).build(options='')

    # in_coords, out_counts = make_coords()
    # real_coords = np.real(in_coords).copy()
    # imag_coords = np.imag(in_coords).copy()

    # gpu_real = cl.Buffer(context, cl.mem_flags.READ_ONLY, real_coords.size * 4)
    # gpu_imag = cl.Buffer(context, cl.mem_flags.READ_ONLY, real_coords.size * 4)
    # gpu_counts = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, real_coords.size * 4)

    # local_size = (8, 8)  # 64 pixels per work group
    # global_size = tuple([round_up(g, l) for g, l in zip(in_coords.shape[::-1], local_size)])
    # width = np.int32(in_coords.shape[1])
    # height = np.int32(in_coords.shape[0])
    # max_iters = np.int32(1024)

    # cl.enqueue_copy(queue, gpu_real, real_coords, is_blocking=False)
    # cl.enqueue_copy(queue, gpu_imag, imag_coords, is_blocking=False)

    # event = program.mandelbrot(queue, global_size, local_size,
    #                            gpu_real, gpu_imag, gpu_counts,
    #                            width, height, max_iters)

    # cl.enqueue_copy(queue, out_counts, gpu_counts, is_blocking=True)

    sampler = GpuLdaSampler(k)

    N_TOPICS = 10
    DOCUMENT_LENGTH = 100
    MAXITER = 50
    P = 2
    FOLDER = "topicimg"

    with Timer() as t:
        # Split docs
        # Split words
        num_workgroups = P
        num_workers = 1

        # Repeat until maxiter or convergence
        for it in range(MAXITER):
            # For P epochs
            for epoch in range(P):
                # For p in range(P)
                # for p in range(P):
                ### Sample
                ## Inputs:
                # (local) m - doc
                # (local) w - word
                # (local) nzw[:,w] - nx|k
                # (local) nz - nk
                # (global) nmz[m,:] - nk|j
                # (global) nm - documents

                ## Outputs
                # p_z - probability vector

                # TODO: Check whether 4 since we might not be passign around ints
                # TODO: Find where to get topics, matrix, etc from
                gpu_topics = cl.Buffer(context, cl.mem_flags.READ_ONLY, self.topics * 4)
                gpu_matrix = cl.Buffer(context, cl.mem_flags.READ_ONLY, self.matrix * 4)
                gpu_p = cl.Buffer(context, cl.mem_flags.READ_ONLY, 4)
                gpu_nzw = cl.Buffer(context, cl.mem_flags.READ_ONLY, self.nzw.size * 4)
                global_nmz = self.nmz
                
                # TODO get these
                alpha = self.alpha
                beta = self.beta

                global_size, local_size = (num_workgroups * num_workers,), (num_workers,)

                gpu_pnz = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, N_TOPICS * 4)
                pnz = []

                cl.enqueue_copy(queue, gpu_topics, self.topics, is_blocking=False)
                cl.enqueue_copy(queue, gpu_matrix, self.matrix, is_blocking=False)
                cl.enqueue_copy(queue, gpu_p, p, is_blocking=False)
                cl.enqueue_copy(queue, gpu_nzw, self.nzw, is_blocking=False)

                event = program.sample(queue, global_size, local_size,
                                        gpu_topics, gpu_matrix, gpu_nzw, gpu_p,
                                        pgu_pnz, global_nmz, alpha, beta)

                cl.enqueue_copy(queue, pnz, gpu_pnz, is_blocking=True)
                seconds = (event.profile.end - event.profile.start) / 1e9
                print 'Sampled in %.3f seconds' % seconds

                # Sync
                with Timer() as t1:
                    sampler._global_update()
                print 'Updated in %.3f seconds' % t1.interval

            print "Iteration {}, run in {} seconds".format(it, t.interval))
            # print "Likelihood", sampler.loglikelihood()

    print 'Fit in %.3f seconds' % t.interval

if __name__ == "__main__":
    import os
    import shutil

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


