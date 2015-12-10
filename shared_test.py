import multiprocessing
import ctypes
import numpy as np
import time


width, height = 10, 400000
shape = (width, height)
#  shared array test inspired by http://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-python-multiprocessing

def make_shared_array():
    shared_array_base = multiprocessing.Array(ctypes.c_double, np.prod(shape))
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())

    return shared_array
def make_shared_array_no_lock():

    shared_array_base = multiprocessing.Array(ctypes.c_double, np.prod(shape), lock=False)
    shared_array = np.ctypeslib.as_array(shared_array_base)

    return shared_array


def make_shared_array_no_lock2():

    return multiprocessing.Array(ctypes.c_double, np.prod(shape), lock=False)

shared_array = make_shared_array_no_lock2()

#-- edited 2015-05-01: the assert check below checks the wrong thing
#   with recent versions of Numpy/multiprocessing. That no copy is made
#   is indicated by the fact that the program prints the output shown below.
## No copy was made
##assert shared_array.base.base is shared_array_base.get_obj()

# Parallel processing
def f(i, def_param=shared_array):
    for w in range(width):
        for h in range(height):
            shared_array[w * height + h] = w + h

def benchmark(f):
    start = time.time()
    pool = multiprocessing.Pool(processes=4)
    pool.map(f, range(width))
    end = time.time()
    print end - start
if __name__ == '__main__':
    # for f in [make_shared_array, make_shared_array_no_lock, make_shared_array_no_lock2]:
    benchmark(f)

