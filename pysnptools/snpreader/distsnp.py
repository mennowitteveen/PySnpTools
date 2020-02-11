import numpy as np
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from pysnptools.snpreader import SnpReader
from pysnptools.snpreader import SnpData


class DistSnp(SnpReader):
    '''
    #!!!cmk22 make this docstring run
    A :class:`.SnpReader` that creates expected SNP values from a :class:`.DistSnp`. No SNP distribution data will be read until
    the :meth:`DistSnp.read` method is called. Use block_size to avoid ever reading all the SNP data into memory.
    at once.

    See :class:`.SnpReader` for general examples of using SnpReaders.

    **Constructor:**
        :Parameters: * **distreader** (:class:`DistReader`) -- The SNP distribution data
                        max_weight=2.0,  #!!!cmk
                     * **block_size** (optional, int) -- The number of SNPs to read at a time.
                     #!!!cmk more

        If **block_size** is not given, then all SNP data will be read at once.

        :Example:

        >>> from __future__ import print_function #Python 2 & 3 compatibility
        >>> dist_on_disk = DistNpz('../examples/toydata.dist.npz')        # A DistNpz file is specified, but nothing is read from disk
        >>> snp_on_disk = DistSnp(snp_on_disk, block_size=50)  # A SnpReader is specified, but nothing is read from disk
        >>> print(dist_on_disk) #Print the specification
        DistSnp(DistNpz('../examples/toydata.dist.npz'),block_size=50)
        >>> print(snp_on_disk.iid_count)                                  # iid information is read from disk, but not SNP data #!!!cmk true?
        500
        >>> snpdata = snp_on_disk.read()                                  # Distribution data is read, 500 at a time, to create an expected SNP value
        >>> print('{0:.6f}'.format(snpdata.val[0,0]))
        0.992307
    '''
    def __init__(self, snpreader, max_weight=2.0, block_size=None):
        super(DistSnp, self).__init__()

        self.distreader = snpreader
        self.max_weight=max_weight
        self.block_size = block_size

    @property
    def row(self):
        return self.distreader.iid

    @property
    def col(self):
        return self.distreader.col

    def __repr__(self):
        return self._internal_repr()

    def _internal_repr(self): #!!! merge this with __repr__
        s = "DistSnp({0}".format(self.distreader)
        if self.block_size is not None:
            s += ",block_size={0}".format(self.block_size)
        s += ")"
        return s

    def copyinputs(self, copier):
        #Doesn't need run_once
        copier.input(self.distreader)

    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        if row_index_or_none is not None or col_index_or_none is not None:
            #!!!cmk22 test this code
            sub = self.distreader[row_index_or_none,col_index_or_none]
        else:
            sub = self.distreader
        return sub._read_snp(max_weight=self.max_weight,block_size=self.block_size, order=order, dtype=dtype, force_python_only=force_python_only, view_ok=view_ok)

    def __getitem__(self, iid_indexer_and_snp_indexer):
        row_index_or_none, col_index_or_none = iid_indexer_and_snp_indexer
        return DistSnp(self.distreader[row_index_or_none,col_index_or_none],max_weight=self.max_weight,block_size=self.block_size)


    @property
    def sid(self):
        '''The :attr:`.SnpReader.sid` property of the SNP data.
        '''
        return self.distreader.sid

    @property
    def sid_count(self):
        '''The :attr:`.SnpReader.sid_count` property of the SNP data.
        '''
        return self.distreader.sid_count

    @property
    def pos(self):
        '''The :attr:`.SnpReader.pos` property of the SNP data.
        '''
        return self.distreader.pos

    def read_snps(self, order='F', dtype=np.float64, force_python_only=False, view_ok=False):
        """Reads the distribution of values and returns a :class:`.SnpData`.

        :param order: {'F' (default), 'C', 'A'}, optional -- Specify the order of the ndarray. If order is 'F' (default),
            then the array will be in F-contiguous order (iid-index varies the fastest).
            If order is 'C', then the returned array will be in C-contiguous order (sid-index varies the fastest).
            If order is 'A', then the :attr:`.SnpData.val`
            ndarray may be in any order (either C-, Fortran-contiguous).
        :type order: string or None

        :param dtype: {scipy.float64 (default), scipy.float32}, optional -- The data-type for the :attr:`.SnpData.val` ndarray.
        :type dtype: data-type

        :param force_python_only: optional -- If False (default), may use outside library code. If True, requests that the read
            be done without outside library code.
        :type force_python_only: bool

        :param view_ok: optional -- If False (default), allocates new memory for the :attr:`.SnpData.val`'s ndarray. If True,
            if practical and reading from a :class:`SnpData`, will return a new 
            :class:`SnpData` with a ndarray shares memory with the original :class:`SnpData`.
            Typically, you'll also wish to use "order='A'" to increase the chance that sharing will be possible.
            Use these parameters with care because any change to either ndarray (for example, via :meth:`.SnpData.standardize`) will effect
            the others. Also keep in mind that :meth:`read` relies on ndarray's mechanisms to decide whether to actually
            share memory and so it may ignore your suggestion and allocate a new ndarray anyway.
        :type view_ok: bool

        :rtype: :class:`.SnpData`

        """
        #!!!where is max_weight?????
        distdata = self.distreader.read(order=order, dtype=dtype, force_python_only=force_python_only, view_ok=view_ok)
        #val /= val.sum(axis=2,keepdims=True)  #make probabilities sum to 1
        print('!!!cmk22 need code')
        return distdata

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
    # There is also a unit test case in 'pysnptools\test.py' that calls this doc test
