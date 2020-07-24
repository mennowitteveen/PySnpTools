from __future__ import absolute_import
from __future__ import print_function
import numpy as np
from itertools import *
import pandas as pd
import logging
from pysnptools.snpreader import SnpReader
from pysnptools.snpreader import SnpData
import math
import warnings
from pysnptools.pstreader import PstData
from six.moves import range

class Bed(SnpReader):
    '''
    A :class:`.SnpReader` for random-access reads of Bed/Bim/Fam files from disk.

    See :class:`.SnpReader` for details and examples.

    The format is described `here <http://zzz.bwh.harvard.edu/plink/binary.shtml>`__.

    **Constructor:**
        :Parameters: * **filename** (*string*) -- The \*.bed file to read. The '.bed' suffix is optional. The related \*.bim and \*.fam files will also be read.
                     * **count_A1** (*bool*) -- Tells if it should count the number of A1 alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.

                     *The following options are never needed, but can be used to avoid reading large '.fam' and '.bim' files when their information is already known.*

                     * **iid** (an array of strings) -- The :attr:`SnpReader.iid` information. If not given, reads info from '.fam' file.
                     * **sid** (an array of strings) -- The :attr:`SnpReader.sid` information. If not given, reads info from '.bim' file.
                     * **pos** (optional, an array of strings) -- The :attr:`SnpReader.pos` information.  If not given, reads info from '.bim' file.
                     * **skip_format_check** (*bool*) -- If False (default), will check that '.bed' file has expected starting bytes.

    **Methods beyond** :class:`.SnpReader`

        The :meth:`.SnpReader.read` method returns a :class:`SnpData` with a :attr:`SnpData.val` ndarray. By default, this ndarray is
        numpy.float32. Optionally, it can be numpy.float16. For :class:`Bed`, however, it can also be numpy.int8 with missing values
        represented by -127.

        :Example:

        >>> from pysnptools.snpreader import Bed
        >>> from pysnptools.util import example_file # Download and return local file name
        >>> bedfile = example_file("tests/datasets/distributed_bed_test1_X.*","*.bed")
        >>> snp_on_disk = Bed(bedfile, count_A1=False)
        >>> snpdata1 = snp_on_disk.read() # Read into the default, an float64 ndarray
        >>> snpdata1.val.dtype
        dtype('float64')
        >>> snpdata1 = snp_on_disk.read(dtype='int8',_require_float32_64=False) #Read into an 'int8' ndarray.
        >>> snpdata1.val.dtype
        dtype('int8')

    '''

    def __init__(self, filename, count_A1=None, iid=None, sid=None, pos=None, skip_format_check=False): #!!!document these new optionals. they are here
        super(Bed, self).__init__()

        self._ran_once = False
        self._file_pointer = None

        self.filename = filename
        if count_A1 is None:
             warnings.warn("'count_A1' was not set. For now it will default to 'False', but in the future it will default to 'True'", FutureWarning)
             count_A1 = False
        self.count_A1 =count_A1
        self.skip_format_check = skip_format_check
        if iid is not None:
            self._row = PstData._fixup_input(iid,empty_creator=lambda ignore:np.empty([0,2],dtype='str'),dtype='str')
        if sid is not None:
            self._col = PstData._fixup_input(sid,empty_creator=lambda ignore:np.empty([0],dtype='str'),dtype='str')
        if pos is not None:
            self._col_property = PstData._fixup_input(pos,count=len(self._col),empty_creator=lambda count:np.array([[np.nan, np.nan, np.nan]]*count))

    def __repr__(self): 
        return "{0}('{1}',count_A1={2})".format(self.__class__.__name__,self.filename,self.count_A1)

    @property
    def row(self):
        """*same as* :attr:`iid`
        """
        if not hasattr(self,"_row"):
            self._row = SnpReader._read_fam(self.filename,remove_suffix="bed")
        return self._row

    @property
    def col(self):
        """*same as* :attr:`sid`
        """
        if not hasattr(self,"_col"):
            self._col, self._col_property = SnpReader._read_map_or_bim(self.filename,remove_suffix="bed", add_suffix="bim")
        return self._col

    @property
    def col_property(self):
        """*same as* :attr:`pos`
        """
        if not hasattr(self,"_col_property"):
            self._col, self._col_property = SnpReader._read_map_or_bim(self.filename,remove_suffix="bed", add_suffix="bim")
        return self._col_property

    def _open_bed(self):
        bedfile = SnpReader._name_of_other_file(self.filename,"bed","bed")
        self._filepointer = open(bedfile, 'rb')
        mode = self._filepointer.read(2)
        if mode != b'l\x1b': raise Exception('No valid binary BED file')
        mode = self._filepointer.read(1) #\x01 = SNP major \x00 = individual major
        if mode != b'\x01': raise Exception('only SNP-major is implemented')
        logging.info("bed file is open {0}".format(bedfile))

    def _close_bed(self):
        self.__del__()
        self.file_pointer = None

    def _run_once(self):
        if self._ran_once:
            return
        self._ran_once = True

        if not hasattr(self,"_row"):
            self._row = SnpReader._read_fam(self.filename,remove_suffix="bed")

        if not hasattr(self,"_col") or not hasattr(self,"_col_property"):
            self._col, self._col_property = SnpReader._read_map_or_bim(self.filename,remove_suffix="bed", add_suffix="bim")
        self._assert_iid_sid_pos(check_val=False)

        if not self.skip_format_check:
            self._open_bed()
            self._close_bed()

    #LATER should Bed have a "flush"?
    def __del__(self):
        if hasattr(self,'_filepointer') and self._filepointer is not None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
            self._filepointer.close()
            self._filepointer = None

    def copyinputs(self, copier):
        # doesn't need to self.run_once() because only uses original inputs
        copier.input(SnpReader._name_of_other_file(self.filename,remove_suffix="bed", add_suffix="bed"))
        copier.input(SnpReader._name_of_other_file(self.filename,remove_suffix="bed", add_suffix="bim"))
        copier.input(SnpReader._name_of_other_file(self.filename,remove_suffix="bed", add_suffix="fam"))


    @staticmethod
    def write(filename, snpdata, count_A1=False, force_python_only=False):
        """Writes a :class:`SnpData` to Bed format and returns the :class:`.Bed`.

        :param filename: the name of the file to create
        :type filename: string
        :param snpdata: The in-memory data that should be written to disk.
        :type snpdata: :class:`SnpData`
        :param count_A1: Tells if it should count the number of A1 alleles (the PLINK standard) or the number of A2 alleles. False is the current default, but in the future the default will change to True.
        :type count_A1: bool
        :rtype: :class:`.Bed`

        >>> from pysnptools.snpreader import Pheno, Bed
        >>> import pysnptools.util as pstutil
        >>> from pysnptools.util import example_file # Download and return local file name
        >>> pheno_fn = example_file("pysnptools/examples/toydata.phe")
        >>> snpdata = Pheno(pheno_fn).read()         # Read data from Pheno format
        >>> pstutil.create_directory_if_necessary("tempdir/toydata.5chrom.bed")
        >>> Bed.write("tempdir/toydata.5chrom.bed",snpdata,count_A1=False)   # Write data in Bed format
        Bed('tempdir/toydata.5chrom.bed',count_A1=False)
        """

        if isinstance(filename,SnpData) and isinstance(snpdata,str): #For backwards compatibility, reverse inputs if necessary
            warnings.warn("write statement should have filename before data to write", DeprecationWarning)
            filename, snpdata = snpdata, filename 

        if count_A1 is None:
             warnings.warn("'count_A1' was not set. For now it will default to 'False', but in the future it will default to 'True'", FutureWarning)
             count_A1 = False

        SnpReader._write_fam(snpdata, filename, remove_suffix="bed")
        SnpReader._write_map_or_bim(snpdata, filename, remove_suffix="bed", add_suffix="bim")

        bedfile = SnpReader._name_of_other_file(filename,remove_suffix="bed", add_suffix="bed")

        if not force_python_only:
            from pysnptools.snpreader import wrap_plink_parser

            if snpdata.val.flags["C_CONTIGUOUS"]:
                order = "C"
            elif snpdata.val.flags["F_CONTIGUOUS"]:
                order = "F"
            else:
                raise Exception("order not known (not 'F' or 'C')")

            if snpdata.val.dtype == np.float64:
                if order=="F":
                    wrap_plink_parser.writePlinkBedFile2doubleFAAA(bedfile.encode('ascii'), snpdata.iid_count, snpdata.sid_count, count_A1, snpdata.val)
                else:
                    wrap_plink_parser.writePlinkBedFile2doubleCAAA(bedfile.encode('ascii'), snpdata.iid_count, snpdata.sid_count, count_A1, snpdata.val)
            elif snpdata.val.dtype == np.float32:
                if order=="F":
                    wrap_plink_parser.writePlinkBedFile2floatFAAA(bedfile.encode('ascii'), snpdata.iid_count, snpdata.sid_count, count_A1, snpdata.val)
                else:
                    wrap_plink_parser.writePlinkBedFile2floatCAAA(bedfile.encode('ascii'), snpdata.iid_count, snpdata.sid_count, count_A1, snpdata.val)
            else:
                raise Exception("dtype '{0}' not known, only float64 and float32".format(snpdata.val.dtype))
            
        else:
            if not count_A1:
                zero_code = 0b00
                two_code = 0b11
            else:
                zero_code = 0b11
                two_code = 0b00

            with open(bedfile,"wb") as bed_filepointer:
                #see http://zzz.bwh.harvard.edu/plink/binary.shtml
                bed_filepointer.write(bytes(bytearray([0b01101100]))) #magic numbers
                bed_filepointer.write(bytes(bytearray([0b00011011]))) #magic numbers
                bed_filepointer.write(bytes(bytearray([0b00000001]))) #snp major

                for sid_index in range(snpdata.sid_count):
                    if sid_index % 1 == 0:
                        logging.info("Writing snp # {0} to file '{1}'".format(sid_index, filename))

                    col = snpdata.val[:, sid_index]
                    for iid_by_four in range(0,snpdata.iid_count,4):
                        vals_for_this_byte = col[iid_by_four:iid_by_four+4]
                        byte = 0b00000000
                        for val_index in range(len(vals_for_this_byte)):
                            val = vals_for_this_byte[val_index]
                            if val == 0:
                                code = zero_code
                            elif val == 1:
                                code = 0b10 #backwards on purpose
                            elif val == 2:
                                code = two_code
                            elif np.isnan(val):
                                code = 0b01 #backwards on purpose
                            else:
                                raise Exception("Can't convert value '{0}' to BED format (only 0,1,2,NAN allowed)".format(val))
                            byte |= (code << (val_index*2))
                        bed_filepointer.write(bytes(bytearray([byte])))
        logging.info("Done writing " + filename)
        return Bed(filename,count_A1=count_A1)

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        self._run_once()

        if order=='A':
            order='F'
        dtype = np.dtype(dtype)

        assert not hasattr(self, 'ind_used'), "A SnpReader should not have a 'ind_used' attribute"

        iid_count_in = self.iid_count
        sid_count_in = self.sid_count

        if iid_index_or_none is not None:
            iid_count_out = len(iid_index_or_none)
            iid_index = iid_index_or_none
        else:
            iid_count_out = iid_count_in
            iid_index = list(range(iid_count_in))

        if sid_index_or_none is not None:
            sid_count_out = len(sid_index_or_none)
            sid_index = sid_index_or_none
        else:
            sid_count_out = sid_count_in
            sid_index = list(range(sid_count_in))

        if not force_python_only:
            from pysnptools.snpreader import wrap_plink_parser
            val = np.zeros((iid_count_out, sid_count_out), order=order, dtype=dtype)
            bed_fn = SnpReader._name_of_other_file(self.filename,"bed","bed")

            if iid_count_in > 0 and sid_count_in > 0:
                if dtype == np.float64:
                    if order=="F":
                        wrap_plink_parser.readPlinkBedFile2doubleFAAA(bed_fn.encode('ascii'), iid_count_in, sid_count_in, self.count_A1, iid_index, sid_index, val)
                    elif order=="C":
                        wrap_plink_parser.readPlinkBedFile2doubleCAAA(bed_fn.encode('ascii'), iid_count_in, sid_count_in, self.count_A1, iid_index, sid_index, val)
                    else:
                        raise Exception("order '{0}' not known, only 'F' and 'C'".format(order));
                elif dtype == np.float32:
                    if order=="F":
                        wrap_plink_parser.readPlinkBedFile2floatFAAA(bed_fn.encode('ascii'), iid_count_in, sid_count_in, self.count_A1, iid_index, sid_index, val)
                    elif order=="C":
                        wrap_plink_parser.readPlinkBedFile2floatCAAA(bed_fn.encode('ascii'), iid_count_in, sid_count_in, self.count_A1, iid_index, sid_index, val)
                    else:
                        raise Exception("order '{0}' not known, only 'F' and 'C'".format(order));
                elif dtype == np.int8:
                    if order=="F":
                        wrap_plink_parser.readPlinkBedFile2int8FAAA(bed_fn.encode('ascii'), iid_count_in, sid_count_in, self.count_A1, iid_index, sid_index, val)
                    elif order=="C":
                        wrap_plink_parser.readPlinkBedFile2int8CAAA(bed_fn.encode('ascii'), iid_count_in, sid_count_in, self.count_A1, iid_index, sid_index, val)
                    else:
                        raise Exception("order '{0}' not known, only 'F' and 'C'".format(order));
                else:
                    raise Exception("dtype '{0}' not known, only float64 and float32".format(dtype))
            
        else:
            if not self.count_A1:
                byteZero = 0
                byteThree = 2
            else:
                byteZero = 2
                byteThree = 0
            if dtype == np.int8:
                missing = -127
            else:
                missing = np.nan
            # An earlier version of this code had a way to read consecutive SNPs of code in one read. May want
            # to add that ability back to the code. 
            # Also, note that reading with python will often result in non-contiguous memory, so the python standardizers will automatically be used, too.       
            self._open_bed()
            #logging.warn("using pure python plink parser (might be much slower!!)")
            val = np.zeros(((int(np.ceil(0.25*iid_count_in))*4),sid_count_out),order=order, dtype=dtype) #allocate it a little big
            for SNPsIndex, bimIndex in enumerate(sid_index):

                startbit = int(np.ceil(0.25*iid_count_in)*bimIndex+3)
                self._filepointer.seek(startbit)
                nbyte = int(np.ceil(0.25*iid_count_in))
                bytes = np.array(bytearray(self._filepointer.read(nbyte))).reshape((int(np.ceil(0.25*iid_count_in)),1),order='F')

                val[3::4,SNPsIndex:SNPsIndex+1]=byteZero
                val[3::4,SNPsIndex:SNPsIndex+1][bytes>=64]=missing
                val[3::4,SNPsIndex:SNPsIndex+1][bytes>=128]=1
                val[3::4,SNPsIndex:SNPsIndex+1][bytes>=192]=byteThree
                bytes=np.mod(bytes,64)
                val[2::4,SNPsIndex:SNPsIndex+1]=byteZero
                val[2::4,SNPsIndex:SNPsIndex+1][bytes>=16]=missing
                val[2::4,SNPsIndex:SNPsIndex+1][bytes>=32]=1
                val[2::4,SNPsIndex:SNPsIndex+1][bytes>=48]=byteThree
                bytes=np.mod(bytes,16)
                val[1::4,SNPsIndex:SNPsIndex+1]=byteZero
                val[1::4,SNPsIndex:SNPsIndex+1][bytes>=4]=missing
                val[1::4,SNPsIndex:SNPsIndex+1][bytes>=8]=1
                val[1::4,SNPsIndex:SNPsIndex+1][bytes>=12]=byteThree
                bytes=np.mod(bytes,4)
                val[0::4,SNPsIndex:SNPsIndex+1]=byteZero
                val[0::4,SNPsIndex:SNPsIndex+1][bytes>=1]=missing
                val[0::4,SNPsIndex:SNPsIndex+1][bytes>=2]=1
                val[0::4,SNPsIndex:SNPsIndex+1][bytes>=3]=byteThree
            val = val[iid_index,:] #reorder or trim any extra allocation
            if not SnpReader._array_properties_are_ok(val, order, dtype):
                val = val.copy(order=order)
            self._close_bed()

        return val


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import os

    if False: #Look for example Bed files with missing data
        from pysnptools.util._example_file import pysnptools_hashdown
        from pysnptools.util import example_file
        for file in pysnptools_hashdown.walk():
            if file.endswith('.bed'):
                print(file+"?")
                bed_file = example_file(file[:-4]+'.*','*.bed')
                bed = Bed(bed_file)
                snpdata = bed[:1000,:1000].read()
                if not np.all(snpdata.val==snpdata.val):
                    print(bed_file+"!")

    if False:
        from pysnptools.snpreader import Bed
        from pysnptools.util import example_file # Download and return local file name
        #bed_file = example_file('doc/ipynb/all.*','*.bed')
        bed_file = r'F:\backup\carlk4d\data\carlk\cachebio\genetics\onemil\id1000000.sid_1000000.seed0.byiid\iid990000to1000000.bed'
        bed = Bed(bed_file,count_A1=False)
        snpdata1 = bed[:,:1000].read()
        snpdata2 = bed[:,:1000].read(dtype='int8',_require_float32_64=False)
        print(snpdata2)
        snpdata3 = bed[:,:1000].read(dtype='int8',order='C',_require_float32_64=False)
        print(snpdata3)
        snpdata3.val=snpdata3.val.astype('float32')
        snpdata3.val.dtype

    if False:
        from pysnptools.snpreader import Bed, SnpGen
        iid_count = 487409
        sid_count = 5000
        sid_count_max =  5765294 
        sid_batch_size = 50

        sid_batch_count = -(sid_count//-sid_batch_size)
        sid_batch_count_max = -(sid_count_max//-sid_batch_size)
        snpgen = SnpGen(seed=234,iid_count=iid_count,sid_count=sid_count_max)

        for batch_index in range(sid_batch_count):
            sid_index_start = batch_index*sid_batch_size
            sid_index_end = (batch_index+1)*sid_batch_size #what about rounding
            filename = r'd:\deldir\rand\fakeukC{0}x{1}-{2}.bed'.format(iid_count,sid_index_start,sid_index_end)
            if not os.path.exists(filename):
                Bed.write(filename+".temp",snpgen[:,sid_index_start:sid_index_end].read())
                os.rename(filename+".temp",filename)


    if False:
        from pysnptools.snpreader import Pheno, Bed

        filename = r'm:\deldir\New folder (4)\all_chr.maf0.001.N300.bed'
        iid_count = 300
        iid = [['0', 'iid_{0}'.format(iid_index)] for iid_index in range(iid_count)]
        bed = Bed(filename, iid=iid,count_A1=False)
        print(bed.iid_count)

    if False:
        from pysnptools.util import example_file
        pheno_fn = example_file("pysnptools/examples/toydata.phe")

    if False:
        from pysnptools.snpreader import Pheno, Bed
        import pysnptools.util as pstutil
        import os
        print(os.getcwd())
        snpdata = Pheno('../examples/toydata.phe').read()         # Read data from Pheno format
        pstutil.create_directory_if_necessary("tempdir/toydata.5chrom.bed")
        Bed.write("tempdir/toydata.5chrom.bed",snpdata,count_A1=False)   # Write data in Bed format

    import doctest
    doctest.testmod(optionflags=doctest.ELLIPSIS)
    # There is also a unit test case in 'pysnptools\test.py' that calls this doc test
