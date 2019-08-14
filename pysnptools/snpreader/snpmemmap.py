import logging
import os
import numpy as np
import unittest
import pysnptools.util as pstutil
from pysnptools.pstreader import PstData
from pysnptools.pstreader import PstMemMap
from pysnptools.snpreader import SnpReader, SnpData


#!!!cmk update documentation
class SnpMemMap(PstMemMap,SnpReader):
    '''
    A :class:`.SnpReader` for reading \*.snp.memmap memory-mapped files on disk.

    See :class:`.SnpReader` for general examples of using SnpReaders.

    **Constructor:**
        :Parameters: * **filename** (*string*) -- The \*.snp.memmap file to read.
        (There is also an additional \*snp.memmap.npz file with column and row data.)

        :Example:

        >>> from pysnptools.snpreader import SnpMemMap
        >>> data_on_disk = SnpMemMap('../examples/little.snp.memmap')
        >>> np_memmap = data_on_disk.read(view_ok=True).val
        >>> print(type(np_memmap)) # To see how to work with numpy's memmap, see https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
        <class 'numpy.memmap'>

    **Methods beyond** :class:`SnpMemMap` #!!!cmk check all these **Methods beyond**'s

    '''

    #!!!cmk document
    def __init__(self, *args, **kwargs):
        super(SnpMemMap, self).__init__(*args, **kwargs)

    @staticmethod
    def write(filename, snpdata):
        """Writes a :class:`SnpData` to SnpMemMap format.

        :param filename: the name of the file to create
        :type filename: string
        :param pstdata: The in-memory data that should be written to disk.
        :type snpdata: :class:`SnpData`

        >>> from pysnptools.snpreader import SnpData, SnpMemMap
        >>> import pysnptools.util as pstutil
        >>> data1 = SnpData(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'], val=[[0.,2.,0.],[0.,1.,2.]], pos=[[0,0,0],[0,0,0],[0,0,0]])
        >>> pstutil.create_directory_if_necessary("tempdir/tiny.snp.memmap")
        >>> SnpMemMap.write("tempdir/tiny.snp.memmap",data1)      # Write data in PstMemMap format
        SnpMemMap('tempdir/tiny.snp.memmap')
        """
        PstMemMap.write(filename,snpdata)#!!!cmk shouldn't all writers return their reader
        return SnpMemMap(filename)

    #!!!cmk document
    @property
    def snp_data(self):
        return self.pst_data

    def run_once(self):
        if (self._ran_once):
            return
        val = self._run_once_inner()
        self._pst_data = SnpData(iid=self._row,sid=self._col,val=val,pos=self._col_property,name="np.memmap('{0}')".format(self._filename))

    #!!!cmk document

    @staticmethod
    def empty(iid, sid, filename, pos=None,order="F",dtype=np.float64):
        self = SnpMemMap(filename)
        self._ran_once = True

        shape = (len(iid),len(sid))
        logging.info("About to start allocating memmap '{0}'".format(filename))
        val = np.memmap(filename, dtype=dtype, mode="w+", order=order, shape=shape)
        logging.info("Finished allocating memmap '{0}'. Size is {1}".format(filename,os.path.getsize(filename)))

        self._row = PstData._fixup_input(iid,empty_creator=lambda ignore:np.empty([0,2],dtype='S'),dtype='S')
        self._col = PstData._fixup_input(sid,empty_creator=lambda ignore:np.empty([0],dtype='S'),dtype='S')
        self._row_property = PstData._fixup_input(None,count=len(self._row),empty_creator=lambda count:np.empty([count,0],dtype='S'),dtype='S')
        self._col_property = PstData._fixup_input(pos,count=len(self._col),empty_creator=lambda count:np.array([[np.nan, np.nan, np.nan]]*count))
        if np.array_equal(self._row, self._col): #If it's square, mark it so by making the col and row the same object
            self._col = self._row
        self._dtype = dtype
        self._order = order

        self._pst_data = SnpData(iid=iid,sid=sid,val=val,pos=pos,name="np.memmap('{0}')".format(filename))
        PstMemMap._write_npz(self._filename, self._pst_data)

        return self

class TestSnpMemMap(unittest.TestCase):     

    def test1(self):
        logging.info("in TestSnpMemMap test1")
        snpreader = SnpMemMap('../examples/little.snp.memmap')
        assert snpreader.iid_count == 300
        assert snpreader.sid_count == 1015
        assert isinstance(snpreader.snp_data.val,np.memmap)

        snpdata = snpreader.read(view_ok=True)
        assert isinstance(snpdata.val,np.memmap)

        filename2 = "tempdir/tiny.snp.memmap"
        pstutil.create_directory_if_necessary(filename2)
        snpreader2 = SnpMemMap.empty(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],filename=filename2,order="F",dtype=np.float64)
        assert isinstance(snpreader2.snp_data.val,np.memmap)
        snpreader2.snp_data.val[:,:] = [[0.,2.,0.],[0.,1.,2.]]
        assert np.array_equal(snpreader2[[1],[1]].read(view_ok=True).val,np.array([[1.]]))
        snpreader2.close()
        assert isinstance(snpreader2.snp_data.val,np.memmap)
        assert np.array_equal(snpreader2[[1],[1]].read(view_ok=True).val,np.array([[1.]]))
        snpreader2.close()

        snpreader3 = SnpMemMap(filename2)
        assert np.array_equal(snpreader3[[1],[1]].read(view_ok=True).val,np.array([[1.]]))
        assert isinstance(snpreader3.snp_data.val,np.memmap)

        #!!!cmk make sure this test gets run

def getTestSuite():
    """
    set up composite test suite
    """
    
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSnpMemMap))
    return test_suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True) #!!!cmk
    r.run(suites)


    import doctest
    doctest.testmod()