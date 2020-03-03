from __future__ import print_function

import logging
import os
import numpy as np
import unittest
import doctest
import pysnptools.util as pstutil
from pysnptools.pstreader import PstData
from pysnptools.pstreader import PstMemMap
from pysnptools.distreader import DistReader, DistData
from pysnptools.util import log_in_place
from os import remove
from shutil import move

class DistMemMap(PstMemMap,DistData):
    '''
    A :class:`.DistData` that keeps its data in a memory-mapped file. This allows data large than fits in main memory.

    See :class:`.DistData` for general examples of using DistData.

    **Constructor:**
        :Parameters: **filename** (*string*) -- The *\*.dist.memmap* file to read.
        
        Also see :meth:`.DistMemMap.empty` and :meth:`.DistMemMap.write`.

        :Example:

        >>> from __future__ import print_function #Python 2 & 3 compatibility
        >>> from pysnptools.distreader import DistMemMap
        >>> dist_mem_map = DistMemMap('../examples/tiny.dist.memmap')
        >>> print(dist_mem_map.val[0,1], dist_mem_map.iid_count, dist_mem_map.sid_count)
        [0.43403135 0.28289911 0.28306954] 25 10

    **Methods inherited from** :class:`.DistData`

        :meth:`.DistData.allclose`, :meth:`.DistData.standardize`

    **Methods beyond** :class:`.DistReader`

    '''

    def __init__(self, *args, **kwargs):
        super(DistMemMap, self).__init__(*args, **kwargs)

    val = property(PstMemMap._get_val,PstMemMap._set_val)
    """The 2D NumPy memmap array of floats that represents the values.

    >>> from pysnptools.distreader import DistMemMap
    >>> dist_mem_map = DistMemMap('../examples/tiny.dist.memmap')
    >>> print(dist_mem_map.val[0,1])
    2.0
    """

    @property
    def offset(self):
        '''The byte position in the file where the memory-mapped values start.
       
        (The disk space before this is used to store :attr:`.DistReader.iid`, etc. information.
        This property is useful when interfacing with, for example, external Fortran and C matrix libraries.)
        
        '''
        self._run_once()
        return self._offset

    @property
    def filename(self):
        '''The name of the memory-mapped file
        '''
        #Don't need '_run_once'
        return self._filename

    @staticmethod
    def empty(iid, sid, filename, pos=None,order="F",dtype=np.float64):
        '''Create an empty :class:`.DistMemMap` on disk.

        :param iid: The :attr:`.DistReader.iid` information
        :type iid: an array of string pairs

        :param sid: The :attr:`.DistReader.sid` information
        :type sid: an array of strings

        :param filename: name of memory-mapped file to create
        :type filename: string

        :param pos: optional -- The additional :attr:`.DistReader.pos` information associated with each sid. Default: None
        :type pos: an array of numeric triples

        :param order: {'F' (default), 'C'}, optional -- Specify the order of the ndarray.
        :type order: string or None

        :param dtype: {scipy.float64 (default), scipy.float32}, optional -- The data-type for the :attr:`.DistMemMap.val` ndarray.
        :type dtype: data-type

        :rtype: :class:`.DistMemMap`

        >>> import pysnptools.util as pstutil
        >>> from pysnptools.distreader import DistMemMap
        >>> filename = "tempdir/tiny.dist.memmap"
        >>> pstutil.create_directory_if_necessary(filename)
        >>> dist_mem_map = DistMemMap.empty(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],filename=filename,order="F",dtype=np.float64)
        >>> dist_mem_map.val[:,:,:] = [[[.5,.5,0],[0,0,1],[.5,.5,0]],
        ...                            [[0,1.,0],[0,.75,.25],[.5,.5,0]]]
        >>> dist_mem_map.flush()

        '''

        self = DistMemMap(filename)
        self._empty_inner(row=iid, col=sid, filename=filename, row_property=None, col_property=pos,order=order,dtype=dtype,val_count=3)
        return self

    def flush(self):
        '''Flush :attr:`.DistMemMap.val` to disk and close the file. (If values or properties are accessed again, the file will be reopened.)

        >>> import pysnptools.util as pstutil
        >>> from pysnptools.distreader import DistMemMap
        >>> filename = "tempdir/tiny.dist.memmap"
        >>> pstutil.create_directory_if_necessary(filename)
        >>> dist_mem_map = DistMemMap.empty(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],filename=filename,order="F",dtype=np.float64)
        >>> dist_mem_map.val[:,:,:] = [[[.5,.5,0],[0,0,1],[.5,.5,0]],
        ...                            [[0,1.,0],[0,.75,.25],[.5,.5,0]]]
        >>> dist_mem_map.flush()

        '''
        if self._ran_once:
            self.val.flush()
            del self._val
            self._ran_once = False


    @staticmethod
    def write(filename, distreader, sid_batch_size=100, dtype=None, order='A'):
        """Writes a :class:`DistData` to :class:`DistMemMap` format. #!!!cmk update for reader and fix up snpmemmap.write etc also

        :param filename: the name of the file to create
        :type filename: string
        :param distdata: The in-memory data that should be written to disk.
        :type distdata: :class:`DistData`
        :rtype: :class:`.DistMemMap`

        >>> import pysnptools.util as pstutil
        >>> from pysnptools.distreader import DistData, DistMemMap
        >>> data1 = DistData(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],
        ...                     val=[[[.5,.5,0],[0,0,1],[.5,.5,0]],
        ...                          [[0,1.,0],[0,.75,.25],[.5,.5,0]]])
        >>> pstutil.create_directory_if_necessary("tempdir/tiny.pst.memmap")
        >>> DistMemMap.write("tempdir/tiny.dist.memmap",data1)      # Write data1 in DistMemMap format
        DistMemMap('tempdir/tiny.dist.memmap')

        """

        #We write iid and sid in ascii for compatibility between Python 2 and Python 3 formats.
        row_ascii = np.array(distreader.row,dtype='S') #!!!avoid this copy when not needed
        col_ascii = np.array(distreader.col,dtype='S') #!!!avoid this copy when not needed

        if hasattr(distreader,'val'):
            order = PstMemMap._order(distreader) if order=='A' else order
            dtype = dtype or distreader.val.dtype
        else:
            order = 'F' if order=='A' else order
            dtype = dtype or np.float64

        self = PstMemMap.empty(row_ascii, col_ascii, filename+'.temp', row_property=distreader.row_property, col_property=distreader.col_property,order=order,dtype=dtype, val_count=3)
        if hasattr(distreader,'val'):
            self.val[:,:,:] = distreader.val
        else:
            start = 0
            with log_in_place("sid_index ", logging.INFO) as updater:
                while start < distreader.sid_count:
                    updater('{0} of {1}'.format(start,distreader.sid_count))
                    distdata = distreader[:,start:start+sid_batch_size].read(order=order,dtype=dtype)
                    self.val[:,start:start+distdata.sid_count,:] = distdata.val
                    start += distdata.sid_count

        self.flush()
        if os.path.exists(filename):
           remove(filename) 
        move(filename+'.temp',filename)#!!!cmk23 do this in snpmemmap, dstmemmap too
        logging.debug("Done writing " + filename) #!!!cmk23 test might be good to write to *.temp and then rename the (here and other place)
        return DistMemMap(filename)



    def _run_once(self):
            if (self._ran_once):
                return
            row_ascii,col_ascii,val,row_property,col_property = self._run_once_inner()
            row = np.array(row_ascii,dtype='str') #!!!avoid this copy when not needed
            col = np.array(col_ascii,dtype='str') #!!!avoid this copy when not needed

            DistData.__init__(self,iid=row,sid=col,val=val,pos=col_property,name="np.memmap('{0}')".format(self._filename))

class TestDistMemMap(unittest.TestCase):     

    def test1(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        filename2 = "tempdir/tiny.dist.memmap"
        pstutil.create_directory_if_necessary(filename2)
        distreader2 = DistMemMap.empty(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],filename=filename2,order="F",dtype=np.float64)
        assert isinstance(distreader2.val,np.memmap)
        distreader2.val[:,:,:] = [[[.5,.5,0],[0,0,1],[.5,.5,0]],[[0,1.,0],[0,.75,.25],[.5,.5,0]]]
        assert np.array_equal(distreader2[[1],[1]].read(view_ok=True).val,np.array([[[0,.75,.25]]]))
        distreader2.flush()
        assert isinstance(distreader2.val,np.memmap)
        assert np.array_equal(distreader2[[1],[1]].read(view_ok=True).val,np.array([[[0,.75,.25]]]))
        distreader2.flush()

        distreader3 = DistMemMap(filename2)
        assert np.array_equal(distreader3[[1],[1]].read(view_ok=True).val,np.array([[[0,.75,.25]]]))
        assert isinstance(distreader3.val,np.memmap)

        logging.info("in TestDistMemMap test1")
        distreader = DistMemMap('../examples/tiny.dist.memmap')
        assert distreader.iid_count == 25
        assert distreader.sid_count == 10
        assert isinstance(distreader.val,np.memmap)

        distdata = distreader.read(view_ok=True)
        assert isinstance(distdata.val,np.memmap)
        os.chdir(old_dir)


def getTestSuite():
    """
    set up composite test suite
    """
    
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDistMemMap))
    return test_suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARN)

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True)
    r.run(suites)

    result = doctest.testmod(optionflags=doctest.ELLIPSIS|doctest.NORMALIZE_WHITESPACE)
    assert result.failed == 0, "failed doc test: " + __file__
