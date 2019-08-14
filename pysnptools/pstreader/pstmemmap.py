import logging
import numpy as np
from pysnptools.pstreader import PstReader, PstData
import unittest
import pysnptools.util as pstutil
import os


class PstMemMap(PstData): #!!!cmk confirm that this doctest gets evaled in testing
    '''
    A :class:`.PstData` that keeps its data in a memory-mapped file.

    See :class:`.PstData` for general examples of using PstData.

    **Constructor:**
        :Parameters: * **filename** (*string*) -- The \*pst.memmap file to read.
        *Also see :meth:`.PstMemMap.empty`

        :Example:

        >>> from pysnptools.pstreader import PstMemMap
        >>> pst_mem_map = PstMemMap('../examples/little.pst.memmap')
        >>> print(pst_mem_map.val[0,1], pstdata.row_count, pstdata.col_count)
        2.0 2 3

    **Methods beyond** :class:`PstMemMap` #!!!cmk check all these **Methods beyond**'s

    '''



    def __init__(self, filename):
        PstReader.__init__(self)
        self._ran_once = False
        self._filename = filename

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self._filename)

    def _get_val(self):#!!!cmk document
        return self._val

    def _set_val(self, new_value):
        if self._val is new_value:
            return
        raise Exception("PstMemMap val's cannot be set to a different array")

    val = property(_get_val,_set_val)


        


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

    @property
    def offset(self):
        '''The byte position in the file where the memory-mapped values start.
        '''
        self.run_once()
        return self._offset

    @property
    def filename(self):
        '''The name of the memory-mapped file
        '''
        #Don't need 'run_once'
        return self._filename

    #!!!cmk document

    @staticmethod
    def empty(row, col, filename, row_property=None, col_property=None,order="F",dtype=np.float64):
        '''Create an empty PstMemMap on disk.

        :param row:  The :attr:`.row` information #!!!cmk finish documenting
        :type standardizer: :class:`.Standardizer`

        :param col: optional -- Deprecated.
        :type block_size: None

        :param filename: If true, returns a second value containing a constant :class:`.Standardizer` trained on this data.
        :type return_trained: boolean

        :param row_property: optional -- If true, will use pure Python instead of faster C++ libraries.
        :type force_python_only: bool

        :param col_property: optional -- If true, will use pure Python instead of faster C++ libraries.
        :type force_python_only: bool

        :param order: optional -- If true, will use pure Python instead of faster C++ libraries.
        :type force_python_only: bool


        :param dtype: optional -- If true, will use pure Python instead of faster C++ libraries.
        :type force_python_only: bool


        :rtype: :class:`.PstMemMap`

        >>> from pysnptools.snpreader import Bed
        >>> snp_on_disk = Bed('../../tests/datasets/all_chr.maf0.001.N300',count_A1=False) # Specify some data on disk in Bed format
        >>> snpdata1 = snp_on_disk.read() # read all SNP values into memory
        >>> print(snpdata1) # Prints the specification for this SnpData
        SnpData(Bed('../../tests/datasets/all_chr.maf0.001.N300',count_A1=False))
        >>> print(snpdata1.val[0,0])
        2.0
        >>> snpdata1.standardize() # standardize changes the values in snpdata1.val and changes the specification.
        SnpData(Bed('../../tests/datasets/all_chr.maf0.001.N300',count_A1=False),Unit())
        >>> print('{0:.6f}'.format(snpdata1.val[0,0]))
        0.229416
        >>> snpdata2 = snp_on_disk.read().standardize() # Read and standardize in one expression with only one ndarray allocated.
        >>> print('{0:.6f}'.format(snpdata2.val[0,0]))
        0.229416

        :Parameters: * **row** (an array of anything) -- The :attr:`.row` information
                     * **col** (an array of anything) -- The :attr:`.col` information
                     * **val** (a 2-D array of floats) -- The values
                     * **row_property** (optional, an array of anything) -- Additional information associated with each row.
                     * **col_property** (optional, an array of strings) -- Additional information associated with each col.
                     * **name** (optional, string) -- Information to be display about the origin of this data
                     * **copyinputs_function** (optional, function) -- *Used internally by optional clustering code*

        '''

        self = PstMemMap(filename)
        self._empty_inner(row, col, filename, row_property, col_property,order,dtype)
        return self


    def _empty_inner(self, row, col, filename, row_property, col_property,order,dtype):
        self._ran_once = True
        self._dtype = dtype
        self._order = order

        row = PstData._fixup_input(row)
        col = PstData._fixup_input(col)
        row_property = PstData._fixup_input(row_property,count=len(row))
        col_property = PstData._fixup_input(col_property,count=len(col))

        with open(filename,'wb') as fp:
            np.save(fp, row)
            np.save(fp, col)
            np.save(fp, row_property)
            np.save(fp, col_property)
            np.save(fp, np.array([self._dtype]))
            np.save(fp, np.array([self._order]))
            self._offset = fp.tell()

        logging.info("About to start allocating memmap '{0}'".format(filename))
        val = np.memmap(filename, offset=self._offset, dtype=dtype, mode="r+", order=order, shape=(len(row),len(col)))
        logging.info("Finished allocating memmap '{0}'. Size is {1}".format(filename,os.path.getsize(filename)))
        PstData.__init__(self,row,col,val,row_property,col_property,name="np.memmap('{0}')".format(filename))

    def run_once(self):
        if (self._ran_once):
            return
        row,col,val,row_property,col_property = self._run_once_inner()
        PstData.__init__(self,row,col,val,row_property,col_property,name="np.memmap('{0}')".format(self._filename))


    def _run_once_inner(self):
        self._ran_once = True

        logging.debug("np.load('{0}')".format(self._filename))
        with open(self._filename,'rb') as fp:
            row = np.load(fp)
            col = np.load(fp)
            row_property = np.load(fp)
            col_property = np.load(fp)
            self._dtype = np.load(fp)[0]
            self._order = np.load(fp)[0]
            self._offset = fp.tell()
        val = np.memmap(self._filename, offset=self._offset, dtype=self._dtype, mode='r', order=self._order, shape=(len(row),len(col)))
        return row,col,val,row_property,col_property

        

    def copyinputs(self, copier):
        # doesn't need to self.run_once()
        copier.input(self._filename)

    # Most _read's support only indexlists or None, but this one supports Slices, too.
    _read_accepts_slices = True
    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        assert view_ok, "Expect view_ok to be True" #!!!cmk good assert?
        self.run_once()
        val, _ = self._apply_sparray_or_slice_to_val(self.val, row_index_or_none, col_index_or_none, self._order, self._dtype, force_python_only) #!!!cmk must confirm that this doesn't copy of view_ok
        return val

    @staticmethod
    def _order(pstdata):
        if pstdata.val.flags['F_CONTIGUOUS']:
            return "F"
        if pstdata.val.flags['C_CONTIGUOUS']:
            return "C"
        raise Exception("Don't know order of PstData's value")



    def flush(self):
        '''Flush the *numpy.memmap* and close the file. (If accessed again, it will be reopened.)
        '''
        if self._ran_once:
            self.val.flush()
            del self._val
            self._ran_once = False



    @staticmethod
    def write(filename, pstdata):
        """Writes a :class:`PstData` to PstMemMap format.

        :param filename: the name of the file to create
        :type filename: string
        :param pstdata: The in-memory data that should be written to disk.
        :type pstmemmap: :class:`PstMemMap`

        >>> from pysnptools.pstreader import PstData, PstMemMap
        >>> import pysnptools.util as pstutil
        >>> data1 = PstData(row=['a','b','c'],col=['y','z'],val=[[1,2],[3,4],[np.nan,6]],row_property=['A','B','C'])
        >>> pstutil.create_directory_if_necessary("tempdir/tiny.pst.memmap")
        >>> PstMemMap.write("tempdir/tiny.pst.memmap",data1)      # Write data1 in PstMemMap format
        PstMemMap('tempdir/tiny.pst.memmap')
        """

        self = PstMemMap.empty(pstdata.row, pstdata.col, filename, row_property=pstdata.row_property, col_property=pstdata.col_property,order=PstMemMap._order(pstdata),dtype=pstdata.val.dtype)
        self.val[:,:] = pstdata.val
        self.flush()
        logging.debug("Done writing " + filename)

        return self


class TestPstMemMap(unittest.TestCase):     

    def test1(self):
        logging.info("in TestPstMemMap test1")

        filename2 = "tempdir/tiny.pst.memmap"
        pstutil.create_directory_if_necessary(filename2)
        pstreader2 = PstMemMap.empty(row=['a','b','c'],col=['y','z'],filename=filename2,row_property=['A','B','C'],order="F",dtype=np.float64)
        assert isinstance(pstreader2.val,np.memmap)
        pstreader2.val[:,:] = [[1,2],[3,4],[np.nan,6]]
        assert np.array_equal(pstreader2[[0],[0]].read(view_ok=True).val,np.array([[1.]]))
        pstreader2.flush()
        assert isinstance(pstreader2.val,np.memmap)
        assert np.array_equal(pstreader2[[0],[0]].read(view_ok=True).val,np.array([[1.]]))
        pstreader2.flush()

        pstreader3 = PstMemMap(filename2)
        assert np.array_equal(pstreader3[[0],[0]].read(view_ok=True).val,np.array([[1.]]))
        assert isinstance(pstreader3.val,np.memmap)

        pstreader = PstMemMap('../examples/tiny.pst.memmap')
        assert pstreader.row_count == 3
        assert pstreader.col_count == 2
        assert isinstance(pstreader.val,np.memmap)

        pstdata = pstreader.read(view_ok=True)
        assert isinstance(pstdata.val,np.memmap)

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

    if False: #!!!cmk make some of this a test
        a=np.ndarray([2,3])
        pointer, read_only_flag = a.__array_interface__['data']
        print pointer
        a*=2
        pointer, read_only_flag = a.__array_interface__['data']
        print pointer
        a = PstMemMap.empty(row=['a','b','c'],col=['y','z'],filename=r'c:\deldir\a.memmap',row_property=['A','B','C'],order="F",dtype=np.float64)
        b = PstData(row=['a','b','c'],col=['y','z'],val=[[1,2],[3,4],[np.nan,6]],row_property=['A','B','C'])
        pointer, read_only_flag = a.val.__array_interface__['data']
        print pointer
        a.val+=1
        a.val+=b.val
        pointer, read_only_flag = a.val.__array_interface__['data']
        print pointer


    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True) #!!!cmk
    r.run(suites)



    import doctest
    #!!!cmk put back: doctest.testmod()
