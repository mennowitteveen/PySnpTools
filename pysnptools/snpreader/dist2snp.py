import numpy as np
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from pysnptools.snpreader import SnpReader
from pysnptools.snpreader import SnpData


class Dist2Snp(SnpReader):
    '''
    A :class:`.SnpReader` that creates expected SNP values from a :class:`.Dist2Snp`. No SNP distribution data will be read until
    the :meth:`Dist2Snp.read` method is called. Use block_size to avoid ever reading all the SNP data into memory.
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
        >>> from pysnptools.distreader import Bgen
        >>> from pysnptools.snpreader import Dist2Snp
        >>> dist_on_disk = Bgen('../examples/2500x100.bgen')        # A DistNpz file is specified, but nothing is read from disk
        >>> snp_on_disk = Dist2Snp(dist_on_disk, block_size=500)  # A SnpReader is specified, but nothing is read from disk
        >>> print(snp_on_disk) #Print the specification
        Dist2Snp(Bgen('../examples/2500x100.bgen'),block_size=500)
        >>> print(snp_on_disk.iid_count)                                  # iid information is read from disk, but not SNP data #!!!cmk true?
        25
        >>> snpdata = snp_on_disk.read()                                  # Distribution data is read, 500 at a time, to create an expected SNP value
        >>> print('{0:.6f}'.format(snpdata.val[0,0]))
        0.776803
    '''
    def __init__(self, snpreader, max_weight=2.0, block_size=None):
        super(Dist2Snp, self).__init__()

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
        s = "Dist2Snp({0}".format(self.distreader)
        if self.block_size is not None:
            s += ",block_size={0}".format(self.block_size)
        s += ")"
        return s

    def copyinputs(self, copier):
        #Doesn't need run_once
        copier.input(self.distreader)

    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        assert row_index_or_none is None and col_index_or_none is None #real assert because indexing should already be pushed to the inner distreader
        return self.distreader._read_snp(max_weight=self.max_weight,block_size=self.block_size, order=order, dtype=dtype, force_python_only=force_python_only, view_ok=view_ok)

    def __getitem__(self, iid_indexer_and_snp_indexer):
        row_index_or_none, col_index_or_none = iid_indexer_and_snp_indexer
        return Dist2Snp(self.distreader[row_index_or_none,col_index_or_none],max_weight=self.max_weight,block_size=self.block_size)


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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
    # There is also a unit test case in 'pysnptools\test.py' that calls this doc test
