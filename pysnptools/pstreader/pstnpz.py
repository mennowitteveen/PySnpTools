import numpy as np
import subprocess, sys, os.path
from itertools import *
import pandas as pd
import logging
from pstreader import PstReader
import pysnptools.util as pstutil

class PstNpz(PstReader):
    '''
    This is a class that reads into memory from PstNpz files.
    '''

    _ran_once = False

    def __init__(self, pstnpz_filename):
        '''
        filename    : string of the name of the npz file.
        '''
        self.pstnpz_filename = pstnpz_filename

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.pstnpz_filename)


    @property
    def row(self):
        self.run_once()
        return self._row

    @property
    def col(self):
        self.run_once()
        return self._col

    @property
    def row_property(self):
        self.run_once()
        return self._row_property

    @property
    def col_property(self):
        self.run_once()
        return self._col_property


    def run_once(self):
        if (self._ran_once):
            return
        self._ran_once = True

        #!!!cmk is this really done without reading 'data'? could mmap support be used?
        with np.load(self.pstnpz_filename) as data: #!! similar code in epistasis
            self._row = data['row']
            self._col = data['col']
            self._row_property = data['row_property']
            self._col_property = data['col_property']
        #!!!cmk??? self._assert_iid_sid_pos()

        return self

    #def __del__(self):
    #    if self._filepointer != None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
    #        self._filepointer.close()

    def copyinputs(self, copier):
        # doesn't need to self.run_once()
        copier.input(self._npz_filename)

    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        if order is None:
            order = "F"
        if dtype is None:
            dtype = np.float64
        if force_python_only is None:
            force_python_only = False

        #This could be re-factored to not use so many names
        row_count_in = self.row_count
        col_count_in = self.col_count

        if row_index_or_none is not None:
            row_count_out = len(row_index_or_none)
            row_index_out = row_index_or_none
        else:
            row_count_out = row_count_in
            row_index_out = range(row_count_in)

        if col_index_or_none is not None:
            col_count_out = len(col_index_or_none)
            col_index_out = col_index_or_none
        else:
            col_count_out = col_count_in
            col_index_out = range(col_count_in)

        with np.load(self.pstnpz_filename) as data: #!! similar code in epistasis
            val = pstutil.sub_matrix(data['val'], row_index_out, col_index_out, order=order, dtype=dtype)

        return val


    @staticmethod
    def write(data, npz_filename):
        logging.info("Start writing " + npz_filename)
        np.savez(npz_filename, row=data.row, col=data.col, row_property=data.row_property, col_property=data.col_property,val=data.val)
        logging.info("Done writing " + npz_filename)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    snpreader = Dat(r'../tests/datasets/all_chr.maf0.001.N300.dat')
    snp_matrix = snpreader.read()
    print len(snp_matrix['sid'])
    snp_matrix = snpreader[:,:].read()
    print len(snp_matrix['sid'])
    sid_index_list = snpreader.sid_to_index(['23_9','23_2'])
    snp_matrix = snpreader[:,sid_index_list].read()
    print ",".join(snp_matrix['sid'])
    snp_matrix = snpreader[:,0:10].read()
    print ",".join(snp_matrix['sid'])

    print snpreader.iid_count
    print snpreader.sid_count
    print len(snpreader.pos)

    snpreader2 = snpreader[::-1,4]
    print snpreader.iid_count
    print snpreader2.sid_count
    print len(snpreader2.pos)

    snp_matrix = snpreader2.read()
    print len(snp_matrix['iid'])
    print len(snp_matrix['sid'])

    snp_matrix = snpreader2[5,:].read()
    print len(snp_matrix['iid'])
    print len(snp_matrix['sid'])

    iid_index_list = snpreader2.iid_to_index(snpreader2.iid[::2])
    snp_matrix = snpreader2[iid_index_list,::3].read()
    print len(snp_matrix['iid'])
    print len(snp_matrix['sid'])

    snp_matrix = snpreader[[4,5],:].read()
    print len(snp_matrix['iid'])
    print len(snp_matrix['sid'])

    print snpreader2
    print snpreader[::-1,4]
    print snpreader2[iid_index_list,::3]
    print snpreader[:,sid_index_list]
    print snpreader2[5,:]
    print snpreader[[4,5],:]
