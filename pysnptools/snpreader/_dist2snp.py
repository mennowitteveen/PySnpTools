import numpy as np
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from pysnptools.snpreader import SnpReader
from pysnptools.snpreader import SnpData


class _Dist2Snp(SnpReader):
    def __init__(self, snpreader, max_weight=2.0, block_size=None):
        super(_Dist2Snp, self).__init__()

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
        s = "{0}.as_snp(".format(self.distreader)
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
        return _Dist2Snp(self.distreader[row_index_or_none,col_index_or_none],max_weight=self.max_weight,block_size=self.block_size)


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
