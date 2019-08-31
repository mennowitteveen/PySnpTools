'''
Runs a distributable job locally in one process. Returns the value of the job.

See SamplePi.py for examples.
'''

from pysnptools.util.mapreduce1.runner import Runner, _run_all_in_memory
import os, sys
import logging

class Local(Runner):
    '''
    A :class:`.Runner` that runs a map_reduce locally. To save memory, it will feed the results of the mapper to the reducer as those results are computed.

    See :func:`.map_reduce` for general examples of use.

    **Constructor:**
        :Parameters: * **mkl_num_threads** (*number*) -- (default None) Limit on the number threads used by the NumPy MKL library.
        :Parameters: * **logging_handler** (*stream*) --  (default stdout) Where to output logging messages.
        
        :Example:

        >>> from pysnptools.util.snpgen import SnpGen
        >>> #Prepare to generate data for 1000 individuals and 1,000,000 SNPs
        >>> snp_gen = SnpGen(seed=332,iid_count=1000,sid_count=1000*1000)
        >>> print snp_gen.iid_count,snp_gen.sid_count
        1000 1000000
        >>> snp_data = snp_gen[:,200*1000:201*1000].read() #Generate for all users and for SNPs 200K to 201K
        >>> print snp_data.val[1,1], snp_data.iid_count, snp_data.sid_count
        0.0 1000 1000

    '''
    def __init__(self, mkl_num_threads = None, logging_handler=logging.StreamHandler(sys.stdout)):
        logger = logging.getLogger()
        if not logger.handlers:
            logger.setLevel(logging.INFO)
        for h in list(logger.handlers):
            logger.removeHandler(h)
        logger.addHandler(logging_handler)
        if logger.level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        
        if mkl_num_threads != None:
            os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)

    def run(self, distributable):
        _JustCheckExists().input(distributable)
        result = _run_all_in_memory(distributable)
        _JustCheckExists().output(distributable)
        return result

class _JustCheckExists(object): #Implements ICopier

    def __init__(self,doPrintOutputNames=False):
        self.doPrintOutputNames = doPrintOutputNames
    
    def input(self,item):
        if isinstance(item, str):
            if not os.path.exists(item): raise Exception("Missing input file '{0}'".format(item))
        elif hasattr(item,"copyinputs"):
            item.copyinputs(self)
        # else -- do nothing

    def output(self,item):
        if isinstance(item, str):
            if not os.path.exists(item): raise Exception("Missing output file '{0}'".format(item))
            if self.doPrintOutputNames:
                print item
        elif hasattr(item,"copyoutputs"):
            item.copyoutputs(self)
        # else -- do nothing
