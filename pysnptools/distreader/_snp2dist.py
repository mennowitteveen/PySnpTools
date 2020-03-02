import numpy as np
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from pysnptools.distreader import DistReader


class _Snp2Dist(DistReader):

    def __init__(self, snpreader, max_weight=2.0, block_size=None):
        super(_Snp2Dist, self).__init__()

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
        s = "{0}.as_dist(".format(self.snpreader)
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
        return _Snp2Dist(self.snpreader[row_index_or_none,col_index_or_none],max_weight=self.max_weight,block_size=self.block_size)


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
