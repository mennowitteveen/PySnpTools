import logging
import numpy as np
from pysnptools.pstreader import PstReader, PstData
import unittest
import pysnptools.util as pstutil
import os


class PstMemMap(PstReader): #!!!cmk confirm that this doctest gets evaled in testing
    '''
    A :class:`.PstReader` for reading \*.pst.memmap memory-mapped files on disk.

    See :class:`.PstReader` for general examples of using PstReaders.

    **Constructor:**
        :Parameters: * **filename** (*string*) -- The \*pst.memmap file to read.
        (There is also an additional \*.pst.memmap.npz file with column and row data.)

        :Example:

        >>> from pysnptools.pstreader import PstMemMap
        >>> data_on_disk = PstMemMap('../examples/little.pst.memmap')
        >>> np_memmap = data_on_disk.read(view_ok=True).val
        >>> print(type(np_memmap)) # To see how to work with numpy's memmap, see https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
        <class 'numpy.memmap'>

    **Methods beyond** :class:`PstMemMap` #!!!cmk check all these **Methods beyond**'s

    '''



    def __init__(self, filename):
        '''
        filename    : string of the name of the memory mapped file.
        (There is also an additional \*.pst.memmap.npz file with column and row data.)
        '''
        #!!!Could (should) there be just one file using memmaps offset feature?
        super(PstMemMap, self).__init__()
        self._ran_once = False
        self._filename = filename

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self._filename)

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

    @property #!!!cmk document
    def pst_data(self):
        self.run_once()
        return self._pst_data


    #!!!cmk document

    @staticmethod
    def empty(row, col, filename, row_property=None, col_property=None,order="F",dtype=np.float64):
        self = PstMemMap(filename)
        self._ran_once = True

        self._row = PstData._fixup_input(row)
        self._col = PstData._fixup_input(col)
        self._row_property = PstData._fixup_input(row_property,count=len(self._row))
        self._col_property = PstData._fixup_input(col_property,count=len(self._col))
        if np.array_equal(self._row, self._col): #If it's square, mark it so by making the col and row the same object
            self._col = self._row
        self._dtype = dtype
        self._order = order

        self._pst_data = PstData(row=row,col=col,val=val,row_property=row_property,col_property=col_property,name="np.memmap('{0}')".format(filename))
        PstMemMap._write_npz(self._filename, self._pst_data)


        shape = (len(row),len(col))
        logging.info("About to start allocating memmap '{0}'".format(filename))
        val = np.memmap(filename, dtype=dtype, mode="w+", order=order, shape=shape)
        logging.info("Finished allocating memmap '{0}'. Size is {1}".format(filename,os.path.getsize(filename)))



        return self


    def run_once(self):
        if (self._ran_once):
            return
        val = self._run_once_inner()
        self._pst_data = PstData(row=self._row,col=self._col,val=val,row_property=self._row_property,col_property=self._col_property,name="np.memmap('{0}')".format(self._filename))

    def _run_once_inner(self):
        self._ran_once = True

        npzfile =self._filename+".npz"
        logging.debug("np.load('{0}')".format(npzfile))
        with np.load(npzfile) as data: #!! similar code in epistasis
            self._row = data['row']
            self._col = data['col']
            if np.array_equal(self._row, self._col): #If it's square, mark it so by making the col and row the same object
                self._col = self._row
            self._row_property = data['row_property']
            self._col_property = data['col_property']
            self._dtype = data['dtype'][0]
            self._order = data['order'][0]

        mode = "r"#!!!cmk understand these modes
        val = np.memmap(self._filename, dtype=self._dtype, mode=mode, order=self._order, shape=(self.row_count,self.col_count))
        return val

        

    def copyinputs(self, copier):
        # doesn't need to self.run_once()
        copier.input(self._filename)
        npzfile = self._filename+'.npz'
        copier.input(npzfile)

    # Most _read's support only indexlists or None, but this one supports Slices, too.
    _read_accepts_slices = True
    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        assert view_ok, "Expect view_ok to be True" #!!! good assert?
        self.run_once()
        val = self._pst_data.val
        val, _ = self._apply_sparray_or_slice_to_val(val, row_index_or_none, col_index_or_none, self._order, self._dtype, force_python_only) #!!! must confirm that this doesn't copy of view_ok
        return val

    @staticmethod
    def _order(pstdata):
        if pstdata.val.flags['F_CONTIGUOUS']:
            return "F"
        if pstdata.val.flags['C_CONTIGUOUS']:
            return "C"
        raise Exception("Don't know order of PstData's value")


    @staticmethod
    def _write_npz(filename, pstdata):
        order = PstMemMap._order(pstdata)
        np.savez(filename, row=pstdata.row, col=pstdata.col, row_property=pstdata.row_property, col_property=pstdata.col_property,dtype=np.array([pstdata.val.dtype]),order=np.array([order]))

    def close(self):
        if self._ran_once:
            self._pst_data.val.flush()
            del self._pst_data.val
            self._ran_once = False



    @staticmethod
    def write(filename, pstdata):
        """Writes a :class:`PstData` to PstMemMap format.

        :param filename: the name of the file to create
        :type filename: string
        :param pstdata: The in-memory data that should be written to disk.
        :type pstdata: :class:`PstData`

        >>> from pysnptools.pstreader import PstData, PstMemMap
        >>> import pysnptools.util as pstutil
        >>> data1 = PstData(row=['a','b','c'],col=['y','z'],val=[[1,2],[3,4],[np.nan,6]],row_property=['A','B','C'])
        >>> pstutil.create_directory_if_necessary("tempdir/tiny.pst.memmap")
        >>> PstMemMap.write("tempdir/tiny.pst.memmap",data1)      # Write data in PstMemMap format
        PstMemMap('tempdir/tiny.pst.memmap')
        """

        self = PstMemMap.empty(pstdata.row, pstdata.col, filename, row_property=pstdata.row_property, col_property=pstdata.col_property,order=PstMemMap._order(pstdata),dtype=pstdata.val.dtype)
        self._pst_data.val[:,:] = pstdata.val
        self.close()
        logging.debug("Done writing " + filename)

        return self


class TestPstMemMap(unittest.TestCase):     

    def test1(self):
        logging.info("in TestPstMemMap test1")
        pstreader = PstMemMap('../examples/little.pst.memmap')
        assert pstreader.row_count == 300
        assert pstreader.col_count == 1015
        assert isinstance(pstreader.pst_data.val,np.memmap)

        pstdata = pstreader.read(view_ok=True)
        assert isinstance(pstdata.val,np.memmap)

        filename2 = "tempdir/tiny.pst.memmap"
        pstutil.create_directory_if_necessary(filename2)
        pstreader2 = PstMemMap.empty(row=['a','b','c'],col=['y','z'],filename=filename2,row_property=['A','B','C'],order="F",dtype=np.float64)
        assert isinstance(pstreader2.pst_data.val,np.memmap)
        pstreader2.pst_data.val[:,:] = [[1,2],[3,4],[np.nan,6]]
        assert np.array_equal(pstreader2[[0],[0]].read(view_ok=True).val,np.array([[1.]]))
        pstreader2.close()
        assert isinstance(pstreader2.pst_data.val,np.memmap)
        assert np.array_equal(pstreader2[[0],[0]].read(view_ok=True).val,np.array([[1.]]))
        pstreader2.close()

        pstreader3 = PstMemMap(filename2)
        assert np.array_equal(pstreader3[[0],[0]].read(view_ok=True).val,np.array([[1.]]))
        assert isinstance(pstreader3.pst_data.val,np.memmap)

        #!!!cmk make sure this test gets run

def getTestSuite():
    """
    set up composite test suite
    """
    
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestPstMemMap))
    return test_suite



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True) #!!!cmk
    r.run(suites)


    if False:
        from pysnptools.pstreader import PstNpz
        data_on_disk = PstNpz('../../tests/datasets/little.pst.npz')
        from pysnptools.pstreader import PstMemMap
        PstMemMap.write("../examples/little.pst.memmap",data_on_disk.read())          # Write data in PstMemMap format

        from pysnptools.pstreader import PstMemMap
        data_on_disk = PstMemMap('../examples/little.pst.memmap')
        np_memmap = data_on_disk.read(view_ok=True).val
        print(type(np_memmap)) # To see how to work with numpy's memmap, see https://docs.scipy.org/doc/numpy/reference/generated/numpy.memmap.html
        #<class 'numpy.memmap'>

        from pysnptools.pstreader import PstData, PstMemMap
        import pysnptools.util as pstutil
        data1 = PstData(row=['a','b','c'],col=['y','z'],val=[[1,2],[3,4],[np.nan,6]],row_property=['A','B','C'])
        pstutil.create_directory_if_necessary("tempdir/tiny.pst.memmap")
        x = PstMemMap.write("tempdir/tiny.pst.memmap",data1)          # Write data in PstMemMap format
        print(x)



    import doctest
    doctest.testmod()
