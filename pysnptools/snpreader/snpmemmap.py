import logging
import os
import shutil
import numpy as np
import unittest
import doctest
from pathlib import Path
import pysnptools.util as pstutil
from pysnptools.pstreader import PstData
from pysnptools.pstreader import PstMemMap
from pysnptools.snpreader import SnpReader, SnpData


class SnpMemMap(PstMemMap,SnpData):
    '''
    A :class:`.SnpData` that keeps its data in a memory-mapped file. This allows data large than fits in main memory.

    See :class:`.SnpData` for general examples of using SnpData.

    **Constructor:**
        :Parameters: **filename** (*string*) -- The *\*.snp.memmap* file to read.
        
        Also see :meth:`.SnpMemMap.empty` and :meth:`.SnpMemMap.write`.

        :Example:

        >>> from pysnptools.snpreader import SnpMemMap
        >>> from pysnptools.util import example_file # Download and return local file name
        >>> mem_map_file = example_file('pysnptools/examples/tiny.snp.memmap')
        >>> snp_mem_map = SnpMemMap(mem_map_file)
        >>> print(snp_mem_map.val[0,1], snp_mem_map.iid_count, snp_mem_map.sid_count)
        2.0 2 3

    **Methods inherited from** :class:`.SnpData`

        :meth:`.SnpData.allclose`, :meth:`.SnpData.standardize`

    **Methods beyond** :class:`.SnpReader`

    '''

    def __init__(self, *args, **kwargs):
        super(SnpMemMap, self).__init__(*args, **kwargs)

    @property
    def val(self):
        """The 2D NumPy memmap array of floats that represents the values. You can get this property, but cannot set it (except with itself)

        >>> from pysnptools.snpreader import SnpMemMap
        >>> from pysnptools.util import example_file # Download and return local file name
        >>> mem_map_file = example_file('pysnptools/examples/tiny.snp.memmap')
        >>> snp_mem_map = SnpMemMap(mem_map_file)
        >>> print(snp_mem_map.val[0,1])
        2.0
        """
        self._run_once()
        return self._val


    @val.setter
    def val(self, new_value):
        self._run_once()
        if self._val is new_value:
            return
        raise Exception("SnpMemMap val's cannot be set to a different array")


    @property
    def offset(self):
        '''The byte position in the file where the memory-mapped values start.
       
        (The disk space before this is used to store :attr:`SnpReader.iid`, etc. information.
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
        '''Create an empty :class:`.SnpMemMap` on disk.

        :param iid: The :attr:`SnpReader.iid` information
        :type iid: an array of string pairs

        :param sid: The :attr:`SnpReader.sid` information
        :type sid: an array of strings

        :param filename: name of memory-mapped file to create
        :type filename: string

        :param pos: optional -- The additional :attr:`SnpReader.pos` information associated with each sid. Default: None
        :type pos: an array of numeric triples

        :param order: {'F' (default), 'C'}, optional -- Specify the order of the ndarray.
        :type order: string or None

        :param dtype: {numpy.float64 (default), numpy.float32}, optional -- The data-type for the :attr:`SnpMemMap.val` ndarray.
        :type dtype: data-type

        :rtype: :class:`.SnpMemMap`

        >>> import pysnptools.util as pstutil
        >>> from pysnptools.snpreader import SnpMemMap
        >>> filename = "tempdir/tiny.snp.memmap"
        >>> pstutil.create_directory_if_necessary(filename)
        >>> snp_mem_map = SnpMemMap.empty(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],filename=filename,order="F",dtype=np.float64)
        >>> snp_mem_map.val[:,:] = [[0.,2.,0.],[0.,1.,2.]]
        >>> snp_mem_map.flush()

        '''

        self = SnpMemMap(filename)
        self._empty_inner(row=iid, col=sid, filename=filename, row_property=None, col_property=pos,order=order,dtype=dtype,val_shape=None)
        return self

    def flush(self):
        '''Flush :attr:`SnpMemMap.val` to disk and close the file. (If values or properties are accessed again, the file will be reopened.)

        >>> import pysnptools.util as pstutil
        >>> from pysnptools.snpreader import SnpMemMap
        >>> filename = "tempdir/tiny.snp.memmap"
        >>> pstutil.create_directory_if_necessary(filename)
        >>> snp_mem_map = SnpMemMap.empty(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],filename=filename,order="F",dtype=np.float64)
        >>> snp_mem_map.val[:,:] = [[0.,2.,0.],[0.,1.,2.]]
        >>> snp_mem_map.flush()

        '''
        if self._ran_once:
            self.val.flush()
            del self._val
            self._ran_once = False


    @staticmethod
    def write(filename, snpdata):
        """Writes a :class:`SnpData` to :class:`SnpMemMap` format.

        :param filename: the name of the file to create
        :type filename: string
        :param snpdata: The in-memory data that should be written to disk.
        :type snpdata: :class:`SnpData`
        :rtype: :class:`.SnpMemMap`

        >>> import pysnptools.util as pstutil
        >>> from pysnptools.snpreader import SnpData, SnpMemMap
        >>> data1 = SnpData(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],val= [[0.,2.,0.],[0.,1.,2.]])
        >>> pstutil.create_directory_if_necessary("tempdir/tiny.snp.memmap") #LATER should we just promise to create directories?
        >>> SnpMemMap.write("tempdir/tiny.snp.memmap",data1)      # Write data1 in SnpMemMap format
        SnpMemMap('tempdir/tiny.snp.memmap')
        """

        #We write iid and sid in ascii for compatibility between Python 2 and Python 3 formats.
        row_ascii = np.array(snpdata.row,dtype='S') #!!!avoid this copy when not needed
        col_ascii = np.array(snpdata.col,dtype='S') #!!!avoid this copy when not needed
        self = PstMemMap.empty(row_ascii, col_ascii, filename+'.temp', row_property=snpdata.row_property, col_property=snpdata.col_property,order=PstMemMap._order(snpdata),dtype=snpdata.val.dtype)
        self.val[:,:] = snpdata.val
        self.flush()
        if os.path.exists(filename):
           os.remove(filename) 
        shutil.move(filename+'.temp',filename)
        logging.debug("Done writing " + filename)
        return SnpMemMap(filename)



    def _run_once(self):
            if (self._ran_once):
                return
            row_ascii,col_ascii,val,row_property,col_property = self._run_once_inner()
            row = np.array(row_ascii,dtype='str') #!!!avoid this copy when not needed
            col = np.array(col_ascii,dtype='str') #!!!avoid this copy when not needed

            SnpData.__init__(self,iid=row,sid=col,val=val,pos=col_property,name="np.memmap('{0}')".format(self._filename))

# !!!cmk make the write method support this
# !!!cmk make fam_fileetc optional
def _bed_to_memmap2(bed_file_list,  memmap_file, fam_file_list, bim_file_list, dtype, start=0,stop=None,step=1000,count_A1=True):
    from pysnptools.snpreader import Bed, _MergeSIDs, SnpMemMap

    memmap_file = Path(memmap_file)
    assert not memmap_file.exists(), f"'{memmap_file}' already exists"

    #######
    # Open the Bed files & construct the properties needed later
    ######
    merge = _MergeSIDs(
        [Bed(bed_file,fam_filename=fam_file,bim_filename=bim_file,count_A1=count_A1,skip_format_check=True)
         for bed_file, fam_file, bim_file in zip(bed_file_list,fam_file_list,bim_file_list)]
        )[:,start:stop]
        
    #######
    # Create a temp file for the memory map
    ######
    memmap_temp = Path(str(memmap_file)+".temp")
    if memmap_temp.exists():
        memmap_temp.unlink()

    memmap = None
    # !!!cmk try:
    memmap = SnpMemMap.empty(iid=merge.iid, sid=merge.sid, filename=str(memmap_temp), pos=merge.pos, dtype=dtype)

    #######
    # In chunks: read, standardize, write
    ######

    # with log_in_place("Bed reader", logging.INFO) as updater:
    for start1 in range(0, merge.sid_count, step):
        logging.info(f"{start1:,} of {merge.sid_count:,}") #!!!cmk logger.INFO or updater
        stop1 = start1+step
        snpdata = merge[:,start1:stop1].read(dtype=dtype)
        snpdata.standardize() #!!!cmk make optional?
        memmap.val[:,start1:stop1] = snpdata.val

    #######
    # If all goes well, flush and rename
    ######
    memmap.flush()
    del memmap
    memmap_temp.rename(memmap_file)
    memmap = None

    ## !!!cmk finally:
    #    if memmap is not None:
    #        memmap.flush()

    #######
    # Return the SnpMemMap reader
    ######

    return SnpMemMap(str(memmap_file))

# !!!cmk make the write method support this
# !!!cmk make fam_fileetc optional
def _bed_to_memmap1(bed_file_list,  memmap_file, fam_file_list, bim_file_list, dtype, step=1000):
    from bed_reader import open_bed
    from pysnptools.snpreader import SnpData, SnpMemMap
    from pysnptools.util import log_in_place

    memmap_file = Path(memmap_file)
    assert not memmap_file.exists(), f"'{memmap_file}' already exists"

    #######
    # Open the Bed files & construct the properties needed later
    ######
    iid = None
    sid_list = []
    pos_list = []
    for bed_file, fam_file, bim_file in zip(bed_file_list,fam_file_list,bim_file_list):
        #!!!cmk add a properties input to avoid reading fields no used
        with open_bed(bed_file,fam_filepath=fam_file, bim_filepath=bim_file) as bed:
            if iid is None:
                iid = np.array([bed.fid,bed.iid]).T
            else:
                assert np.array_equal(iid, np.array([bed.fid,bed.iid]).T), "Expect all fam files to agree"
            sid_list.append(bed.sid)
            pos_list.append(np.array([bed.chromosome.astype("float"),
                                      bed.cm_position,
                                      bed.bp_position]).T)
    assert iid is not None, "Expect a least one bed file"
    sid = np.hstack(sid_list)
    pos = np.vstack(pos_list)
        
    #######
    # Create a temp file for the memory map
    ######
    memmap_temp = Path(str(memmap_file)+".temp")
    if memmap_temp.exists():
        memmap_temp.unlink()

    memmap = None
    try:
        memmap = SnpMemMap.empty(iid=iid, sid=sid, filename=str(memmap_temp), pos=pos, dtype=dtype)

        #######
        # In chunks: read, standardize, write
        ######

        offset = 0
        with log_in_place("Bed reader", logging.INFO) as updater:
            for bed_file, fam_file, bim_file in zip(bed_file_list,fam_file_list,bim_file_list):
                with open_bed(bed_file,fam_filepath=fam_file, bim_filepath=bim_file) as bed:
                    for start1 in range(0, bed.sid_count, step):
                        updater(f"{start1:,} of {bed.sid_count:,} in '{bed_file}") #!!!cmk logger.INFO or updater
                        stop1 = start1+step
                        val = bed.read(np.s_[start1:stop1],dtype=dtype)
                        start2 = start1 + offset
                        stop2 = start2 + val.shape[1] # OK for stop1 go beyond the end of one 1, but shop2 shouldn't to next file
                        snpdata = SnpData(val=val, iid=iid, sid=sid[start2:stop2], pos=pos[start2:stop2,:])
                        snpdata.standardize() #!!!cmk make optional?
                        memmap.val[:,start2:stop2] = snpdata.val
                    offset += bed.sid_count                    

            #######
            # If all goes well, flush and rename
            ######
            memmap.flush() #!!!cmk use a try/catch?
            del memmap
            memmap_temp.rename(memmap_file)
            memmap = None

    finally:
        if memmap is not None:
            memmap.flush()

    #######
    # Return the SnpMemMap reader
    ######

    return SnpMemMap(str(memmap_file))

class TestSnpMemMap(unittest.TestCase):     

    def test1(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        filename2 = "tempdir/tiny.snp.memmap"
        pstutil.create_directory_if_necessary(filename2)
        snpreader2 = SnpMemMap.empty(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],filename=filename2,order="F",dtype=np.float64)
        assert isinstance(snpreader2.val,np.memmap)
        snpreader2.val[:,:] = [[0.,2.,0.],[0.,1.,2.]]
        assert np.array_equal(snpreader2[[1],[1]].read(view_ok=True).val,np.array([[1.]]))
        snpreader2.flush()
        assert isinstance(snpreader2.val,np.memmap)
        assert np.array_equal(snpreader2[[1],[1]].read(view_ok=True).val,np.array([[1.]]))
        snpreader2.flush()

        snpreader3 = SnpMemMap(filename2)
        assert np.array_equal(snpreader3[[1],[1]].read(view_ok=True).val,np.array([[1.]]))
        assert isinstance(snpreader3.val,np.memmap)

        logging.info("in TestSnpMemMap test1")
        snpreader = SnpMemMap('../examples/tiny.snp.memmap')
        assert snpreader.iid_count == 2
        assert snpreader.sid_count == 3
        assert isinstance(snpreader.val,np.memmap)

        snpdata = snpreader.read(view_ok=True)
        assert isinstance(snpdata.val,np.memmap)
        os.chdir(old_dir)


def getTestSuite():
    """
    set up composite test suite
    """
    
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSnpMemMap))
    return test_suite


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    #!!!!cmk
    if True:
        # Data will appear in the memory mapped file in the order given.
        # The *.fam and *.bim files must be in the same order as the bed files.
        bed_file_list = []
        fam_file_list = []
        bim_file_list = []
        for piece in range(25):
             bed_file_list += [r"M:\deldir\testsnps_1_10_250000_10000\chrom10.piece{0}of25.bed".format(piece)]
             fam_file_list += [r"M:\deldir\testsnps_1_10_250000_10000\chrom10.piece{0}of25.fam".format(piece)]
             bim_file_list += [r"M:\deldir\testsnps_1_10_250000_10000\chrom10.piece{0}of25.bim".format(piece)]
        #for chrom in range(21,23):
        #    bed_file_list += [r"d:\deldir\genbgen\merged_487400x220000.{0}.bed".format(chrom)]
        #    fam_file_list += [r"m:\deldir\genbgen\merged_487400x220000.{0}.fam".format(chrom)]
        #    bim_file_list += [r"m:\deldir\genbgen\merged_487400x220000.{0}.bim".format(chrom)]

        memmap_file = r"D:\deldir\memmap1.snp.memmap"

        #######
        # For this demo, erase the memmap output file
        ######

        if Path(memmap_file).exists():
            Path(memmap_file).unlink()

        #######
        # For this demo, create a memmap file
        ######

        memmap = _bed_to_memmap2(bed_file_list,fam_file_list=fam_file_list,bim_file_list=bim_file_list,memmap_file=memmap_file,dtype='float32',step=10)
        memmap

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True)
    ret = r.run(suites)
    assert ret.wasSuccessful()

    result = doctest.testmod(optionflags=doctest.ELLIPSIS)
    assert result.failed == 0, "failed doc test: " + __file__
