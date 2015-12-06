import numpy as np
import scipy as sp
from scipy.special import gammaln
import pyopencl as cl
import time
from copy import copy

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

class GPULdaSampler(object):
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
        self.nzw = np.zeros((self.n_topics, vocab_size))
        self.nm = np.zeros(n_docs)
        self.nz = np.zeros(self.n_topics)
        self.topics = np.zeros((n_docs, vocab_size))

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
        ############################ Setup CL
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

        program = cl.Program(context, open('gpu_lda_gibbs.cl').read()).build(options='')

        ##################### Let's Roll
        n_docs, vocab_size = matrix.shape
        self._initialize(matrix)
        num_workgroups = self.P
        num_workers = 1

        for it in xrange(maxiter):
            # For P epochs
            for epoch in range(self.P):
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

                # Flatten
                flat_matrix = np.ravel(matrix)
                flat_topics = np.ravel(self.topics).astype(np.int32)
                flat_nzw = np.ravel(self.nzw).astype(np.int32)
                flat_nmz = np.ravel(self.nmz).astype(np.int32)
                print flat_matrix.size

                # Input Buffers
                gpu_matrix = cl.Buffer(context, cl.mem_flags.READ_ONLY, flat_matrix.size * 4)
                gpu_topics = cl.Buffer(context, cl.mem_flags.READ_ONLY, flat_topics.size * 4)
                gpu_nzw = cl.Buffer(context, cl.mem_flags.READ_ONLY, flat_nzw.size * 4)
                global_nmz = cl.Buffer(context, cl.mem_flags.READ_ONLY, flat_nmz.size * 4)
                
                # Ints and Floats
                alpha = np.float32(self.alpha)
                beta = np.float32(self.beta)
                n_topics = np.int32(self.n_topics)
                gpu_p = np.int32(epoch)

                # Sizing
                global_size, local_size = (num_workgroups * num_workers,), (num_workers,)

                # Output
                gpu_pnz = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, self.topics.size * 4)
                pnz = np.zeros(self.n_topics)

                # Enqueues
                cl.enqueue_copy(queue, gpu_matrix, flat_matrix, is_blocking=False)
                cl.enqueue_copy(queue, gpu_topics, flat_topics, is_blocking=False)
                cl.enqueue_copy(queue, gpu_nzw, flat_nzw, is_blocking=False)
                cl.enqueue_copy(queue, global_nmz, flat_nmz, is_blocking=False)

                event = program.sample(queue, global_size, local_size,
                                        gpu_topics, gpu_matrix, gpu_nzw, global_nmz,
                                        gpu_pnz, gpu_p, n_topics, alpha, beta)

                cl.enqueue_copy(queue, pnz, gpu_pnz, is_blocking=True)
                seconds = (event.profile.end - event.profile.start) / 1e9
                print 'Sampled in %.3f seconds' % seconds

                # Sync
                start = time.time()
                self._global_update()
                end = time.time()
                print 'Updated in %.3f seconds' % (end - start)
                print "Epoch " + str(epoch) + " finished"

            print "Iteration " + str(it) + " finished" 
            # print "Likelihood", self.loglikelihood()

        print 'Fit in %.3f seconds' % t.interval
        yield 1