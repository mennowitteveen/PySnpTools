import os
import numpy as np
import logging
from pysnptools.snpreader import SnpReader

class Pairs(SnpReader):
    '''
    !!!cmk need to update
    A :class:`.SnpReader` for random-access reads of Bed/Bim/Fam files from disk.

    See :class:`.SnpReader` for details and examples.

    The format is described in http://pngu.mgh.harvard.edu/~purcell/plink/binary.shtml.

    **Constructor:**
        :Parameters: * **filename** (*string*) -- The \*.bed file to read. The '.bed' suffix is optional. The related \*.bim and \*.fam files will also be read.
                     * **count_A1** (*bool*) -- Tells if it should count the number of A1 alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.

                     *The following options are never needed, but can be used to avoid reading large '.fam' and '.bim' files when their information is already known.*

                     * **iid** (an array of strings) -- The :attr:`.SnpReader.iid` information. If not given, reads info from '.fam' file.
                     * **sid** (an array of strings) -- The :attr:`.SnpReader.sid` information. If not given, reads info from '.bim' file.
                     * **pos** (optional, an array of strings) -- The :attr:`.SnpReader.pos` information.  If not given, reads info from '.bim' file.
                     * **skip_format_check** (*bool*) -- If False (default), will check that '.bed' file has expected starting bytes.

    **Methods beyond** :class:`.SnpReader`
    '''

    #!!!cmk see fastlmm\association\epistasis.py for code that allows ranges of snps to be specified when making pairs
    def __init__(self, snpreader,include_singles=False): #!!!cmk could add option to change snp separator and another to encode chrom, etc in the snp name
        super(Pairs, self).__init__()
        self.snpreader = snpreader
        self.include_singles=include_singles

    def __repr__(self): 
        return "{0}({1})".format(self.__class__.__name__,self.snpreader)

    @property
    def row(self):
        """*same as* :attr:`iid`
        """
        return self.snpreader.row

    @property
    def col(self):
        """*same as* :attr:`sid`
        """
        if not hasattr(self,"_col"):
            col_0 = self.snpreader.col
            col_list = []
            self.index0_list = []
            self.index1_list = []
            for index0 in xrange(self.snpreader.col_count):
                logging.info("index0={0} of {1}".format(index0,self.snpreader.col_count))
                start1 = index0 if self.include_singles else index0+1
                for index1 in xrange(start1,self.snpreader.col_count):
                    col_list.append('{0},{1}'.format(col_0[index0],col_0[index1]))
                    self.index0_list.append(index0)
                    self.index1_list.append(index1)
            self._col = np.array(col_list)
            self.index0_list = np.array(self.index0_list)
            self.index1_list = np.array(self.index1_list)

        assert self.col_count == len(self._col), "real assert"
        return self._col

    @property
    def col_count(self):
        n = self.snpreader.col_count
        if self.include_singles:
            return (n*n+n)/2
        else:
            return (n*n-n)/2
    @property
    def col_property(self):
        """*same as* :attr:`pos`
        """
        if not hasattr(self,"_col_property"):
            self._col_property = np.zeros([self.sid_count,3],dtype=np.int64)
        return self._col_property

    def copyinputs(self, copier):
        # doesn't need to self.run_once() because only uses original inputs
        self.snpreader.copyinputs(copier)


    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        iid_count_in = self.iid_count
        sid_count_in = self.sid_count

        if iid_index_or_none is not None:
            iid_count_out = len(iid_index_or_none)
            iid_index_out = iid_index_or_none
        else:
            iid_count_out = iid_count_in
            iid_index_out = range(iid_count_in)

        if sid_index_or_none is not None:
            sid_count_out = len(sid_index_or_none)
            sid_index_out = sid_index_or_none
        else:
            sid_count_out = sid_count_in
            sid_index_out = range(sid_count_in)

        
        sid_index_inner_0 = self.index0_list[sid_index_out] #Find the sid_index of the left snps of interest
        sid_index_inner_1 = self.index1_list[sid_index_out] #Find the sid_index of the right snps of interest
        sid_index_inner_01 = np.unique(np.r_[sid_index_inner_0,sid_index_inner_1]) #Index of every snp of interest
        val_inner_01 = self.snpreader[iid_index_or_none,sid_index_inner_01].read(order=order, dtype=dtype, force_python_only=force_python_only, view_ok=True).val #read every val of interest

        sid_index_inner_01_reverse = {v:i for i,v in enumerate(sid_index_inner_01)} #Dictionary of snp_index to position in sid_index_inner_01
        sid_index_inner_0_in_val = np.array([sid_index_inner_01_reverse[i] for i in sid_index_inner_0])  #Replace snp_index0 with column # in val_inner_01
        sid_index_inner_1_in_val = np.array([sid_index_inner_01_reverse[i] for i in sid_index_inner_1])  #Replace snp_index1 with column # in val_inner_01
        val_inner_0 = val_inner_01[:,sid_index_inner_0_in_val] #Extract the vals for the left snps of interest
        val_inner_1 = val_inner_01[:,sid_index_inner_1_in_val]#Extract the vals for the right snps of interest
        val = val_inner_0*val_inner_1 #Element multiplication creates the vals for the pairs

        return val


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    from pysnptools.snpreader import Pheno, Bed
    import pysnptools.util as pstutil
    #snpdata = Pheno('pysnptools/examples/toydata.phe').read()         # Read data from Pheno format

    #!!!cmk move these to test file
    pairs = Pairs(Bed('pysnptools/examples/toydata.bed',count_A1=False)[:,:100])
    print(pairs.iid)
    print(pairs.sid)
    print(pairs.pos)
    print(pairs.row_property)
    snpdata = pairs[:,10:20].read()
    print(snpdata.val)

    #!!!cmkimport doctest
    #doctest.testmod()
    # There is also a unit test case in 'pysnptools\test.py' that calls this doc test
