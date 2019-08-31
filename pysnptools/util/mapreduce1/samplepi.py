'''

Some examples of IDistributable and Runner. Classes that implement IDistributable specify work to be done.
The class defined in this file, SamplePi, implements IDistributable to approximate  PI by simulating dart throws.

Classes that implement Runner tell how to do that work. Examples of IRunner classes are Local, LocalMultiProc,
LocalRunInParts, and HPC.

Here are examples of each Runner running a SamplePi job (which is defined below):


Run local in a single process

    >>> from pysnptools.util.mapreduce1.samplepi import *
    >>> round(sample_pi(dartboard_count=1000,dart_count=1000,runner=Local()),2)
    pi ~ 3.162
    3.16


Run local on 12 processors (also, increase the # of dartboards and darts)

    >>> from pysnptools.util.mapreduce1.samplepi import *          #LocalMultiProc and HPC won't work without this 'from'
    >>> runner = LocalMultiProc(12,mkl_num_threads=1)
    >>> result = sample_pi(dartboard_count=1000,dart_count=1000,runner=runner)
    >>> round(result,2)
    pi ~ 3.138856
    3.14



'''

from pysnptools.util.mapreduce1.runner import *
from pysnptools.util.mapreduce1.mapreduce import map_reduce
import logging

def sample_pi(dartboard_count,dart_count,runner):
    '''
    Finds an approximation of pi by throwing  darts in a 2 x 2 square and seeing how many land within 1 of the center.
    '''
    def mapper1(work_index):
        import scipy as sp
        from numpy.random import RandomState
        # seed the global random number generator with work_index xor'd with an arbitrary constant
        randomstate = RandomState(work_index ^ 284882)
        sum = 0.0
        for i in xrange(self.dart_count):
            x = randomstate.uniform(2)
            y = randomstate.uniform(2)
            is_in_circle = sp.sqrt((x-1)**2+(y-1)**2) < 1
            if is_in_circle:
                sum += 1
        fraction_in_circle = sum / dart_count
        return fraction_in_circle


    def reducer1(result_sequence):
        average = float(sum(result_sequence)) / dartboard_count
        # the circle has area pi * r ** 2 = pi. the square has area 2**2=4, so the fraction_in_circle ~ pi /4
        pi = average * 4
        print("pi ~ {0}".format(pi))
        return pi

    result = map_reduce(xrange(dartboard_count),mapper1,reducer1,runner=runner)
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
