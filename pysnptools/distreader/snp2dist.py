import numpy as np
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from pysnptools.distreader import DistReader


class Snp2Dist(DistReader):
    '''#!!!cmk22 test this doc string
    A :class:`.DistReader` that creates a distribution from a :class:`.SnpReader`. No SNP data will be read until
    the :meth:`Snp2Dist.read` method is called. Use block_size to avoid ever reading all the SNP data into memory.
    at once.

    See :class:`.DistReader` for general examples of using DistReaders.

    **Constructor:**
        :Parameters: * **snpreader** (:class:`SnpReader`) -- The SNP data
                     * **block_size** (optional, int) -- The number of SNPs to read at a time.
                     #!!!cmk more

        If **block_size** is not given, then all SNP data will be read at once.

        :Example:

        >>> from __future__ import print_function #Python 2 & 3 compatibility
        >>> from pysnptools.snpreader import Bed
        >>> from pysnptools.distreader import Snp2Dist
        >>> snp_on_disk = Bed('../examples/toydata.bed',count_A1=True)     # A Bed file is specified, but nothing is read from disk
        >>> dist_on_disk = Snp2Dist(snp_on_disk, block_size=500)           # A DistReader is specified, but nothing is read from disk
        >>> print(dist_on_disk) #Print the specification
        Snp2Dist(Bed('../examples/toydata.bed',count_A1=True),block_size=500)
        >>> cmkprint(dist_on_disk.iid_count)                                  # iid information is read from disk, but not SNP data
        500
        >>> distdata = dist_on_disk.read()                                 # Snp data is read, 500 at a time, to create distribution values
        >>> print(distdata.val[0,0,:])
        [0. 1. 0.
    '''
    def __init__(self, snpreader, max_weight=2.0, block_size=None):
        super(Snp2Dist, self).__init__()

        self.snpreader = snpreader
        self.max_weight = max_weight
        self.block_size = block_size

    @property
    def row(self):
        return self.snpreader.iid

    @property
    def col(self):
        return self.snpreader.col

    def __repr__(self):
        return self._internal_repr()

    def _internal_repr(self): #!!! merge this with __repr__
        s = "Snp2Dist({0}".format(self.snpreader)
        if self.block_size is not None:
            s += ",block_size={0}".format(self.block_size)
        s += ")"
        return s

    def copyinputs(self, copier):
        #Doesn't need run_once
        copier.input(self.snpreader)

    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        assert row_index_or_none is None and col_index_or_none is None #real assert because indexing should already be pushed to the inner snpreader
        return self.snpreader._read_dist(max_weight=self.max_weight,block_size=self.block_size, order=order, dtype=dtype, force_python_only=force_python_only, view_ok=view_ok)

    def __getitem__(self, iid_indexer_and_snp_indexer):
        row_index_or_none, col_index_or_none = iid_indexer_and_snp_indexer
        return Snp2Dist(self.snpreader[row_index_or_none,col_index_or_none],max_weight=self.max_weight,block_size=self.block_size)


    @property
    def sid(self):
        '''The :attr:`.SnpReader.sid` property of the SNP data.
        '''
        return self.snpreader.sid

    @property
    def sid_count(self):
        '''The :attr:`.SnpReader.sid_count` property of the SNP data.
        '''
        return self.snpreader.sid_count

    @property
    def pos(self):
        '''The :attr:`.SnpReader.pos` property of the SNP data.
        '''
        return self.snpreader.pos


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
    # There is also a unit test case in 'pysnptools\test.py' that calls this doc test
