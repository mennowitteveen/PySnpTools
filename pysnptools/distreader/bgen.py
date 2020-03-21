import os
import logging
import numpy as np
import warnings
import unittest
import sys

if sys.version_info[0] >= 3:
    from bgen_reader import read_bgen
    from bgen_reader import example_files
    from bgen_reader import create_metafile
from tempfile import TemporaryFile
from pysnptools.util import log_in_place
import shutil
import math
import subprocess
import pysnptools.util as pstutil
from pysnptools.distreader import DistReader

def default_iid_function(sample):
    '''
    The default function for turning a Bgen sample into a two-part :attr:`pysnptools.distreader.DistReader.iid`.
    If the Bgen sample contains a single comma, we split on the comma to create the iid.
    Otherwise, the iid will be ('0',sample)

    >>> default_iid_function('fam0,ind0')
    ('fam0', 'ind0')
    >>> default_iid_function('ind0')
    ('0', 'ind0')
    '''
    fields = sample.split(',')
    if len(fields)==2:
        return fields[0],fields[1]
    else:
        return ('0',sample)

def default_sid_function(id,rsid):
    '''
    The default function for turning a Bgen (SNP) id and rsid into a :attr:`pysnptools.distreader.DistReader.sid`.
    If the Bgen rsid is '' or '0', the sid will be the (SNP) id.
    Otherwise, the sid will be 'ID,RSID'

    >>> default_sid_function('SNP1','rs102343')
    'SNP1,rs102343'
    >>> default_sid_function('SNP1','0')
    'SNP1'
    '''
    if rsid=='0' or rsid=='':
        return id
    else:
        return id+','+rsid

def default_sample_function(famid,indid):
    '''
    The default function for turning a two-part :attr:`pysnptools.distreader.DistReader.iid` into a a Bgen sample.
    If the iid's first part (the family id) is '0' or '', the sample will be iid's 2nd part.
    Otherwise, the sample will be 'FAMID,INDID'

    >>> default_sample_function('fam0','ind0')
    'fam0,ind0'
    >>> default_sample_function('0','ind0')
    'ind0'
    '''
    if famid=='0' or famid=='':
        return indid
    else:
        return famid+','+indid

def default_id_rsid_function(sid):
    '''
    The default function for turning a :attr:`pysnptools.distreader.DistReader.sid` into a Bgen (SNP) id and rsid.
    If the sid contains a single comma, we split on the comma to create the id and rsid.
    Otherwise, the (SNP) id will be the sid and the rsid will be '0'

    >>> default_id_rsid_function('SNP1,rs102343')
    ('SNP1', 'rs102343')
    >>> default_id_rsid_function('SNP1')
    ('SNP1', '0')
    '''
    fields = sid.split(',')
    if len(fields)==2:
        return fields[0],fields[1]
    else:
        return sid,'0'



class Bgen(DistReader):
    '''
    A :class:`.DistReader` for reading \*.bgen files from disk.

    See :class:`.DistReader` for general examples of using DistReaders.

    The BGEN format is described `here <https://www.well.ox.ac.uk/~gav/bgen_format/>`__.

        **Tip:** The 'gen' in BGEN stands for 'genetic'. The 'gen' in :class:`DistGen` stands for generate, because it generates random (genetic) data.*
   
    **Constructor:**
        :Parameters: * **filename** (string) -- The BGEN file to read.
                     * **iid_function** (optional, function) -- Function to turn a BGEN sample into a :attr:`DistReader.iid`.
                       (Default: :meth:`bgen.default_iid_function`.)
                     * **sid_function** (optional, function or string) -- Function to turn a BGEN (SNP) id and rsid into a :attr:`DistReader.sid`.
                       (Default: :meth:`bgen.default_sid_function`.) Can also be the string 'id' or 'rsid', which is faster than using a function.
                     * **verbose** (optional, boolean) -- If true, reads verbosely. Defaults to false.
                     * **metadata** (optional, string) -- Name of an existing metadata file. Defaults to '*filename*.sample'.
                       If not given and doesn't exist, will be created.
                     * **sample** (optional, string) -- A GEN sample file. If given, overrides information in \*.bgen file.

        :Example:

        >>> from __future__ import print_function #Python 2 & 3 compatibility
        >>> from pysnptools.distreader import Bgen
        >>> data_on_disk = Bgen('../examples/example.bgen')
        >>> print((data_on_disk.iid_count, data_on_disk.sid_count))
        (500, 199)

    **Methods beyond** :class:`.DistReader`

    '''
    _warning_dictionary = {}
    def __init__(self, filename, iid_function=default_iid_function, sid_function=default_sid_function, verbose=False, metadata=None, sample=None):
        super(Bgen, self).__init__()
        self._ran_once = False
        self.read_bgen = None

        self.filename = filename
        self._verbose = verbose
        self._iid_function = iid_function
        self._sid_function = sid_function
        self._sample = sample
        self._metadata = metadata

    def _metadata_file_name(self):
        if self._metadata is not None:
            return self._metadata
        return self.filename+".metadata"

    def _metadata2_file_name(self):
        sample_hash = '' if self._sample is None else hash(self._sample) #If they give a sample file, we need a different metadata2 file.
        return self.filename + ".metadata{0}.npz".format(sample_hash)


    @property
    def row(self):
        self._run_once()
        return self._row

    @property
    def col(self):
        self._run_once()
        return self._col

    @property
    def col_property(self):
        """*same as* :attr:`pos`
        """
        self._run_once()
        return self._col_property


    def _run_once(self):
        if self._ran_once:
            return
        self._ran_once = True

        assert os.path.exists(self.filename), "Expect file to exist ('{0}')".format(self.filename)

        #LATER remove when bug is fixed
        new_file_date = os.stat(self.filename).st_ctime
        old_file_date = Bgen._warning_dictionary.get(self.filename)
        if old_file_date is not None and old_file_date != new_file_date:
            logging.warning('Opening a file again, but its creation date has changed See https://github.com/limix/bgen-reader-py/issues/25. File "{0}"'.format(self.filename))
        Bgen._warning_dictionary[self.filename] = new_file_date

        self._read_bgen = read_bgen(self.filename,metafile_filepath=self._metadata,samples_filepath=self._sample,verbose=self._verbose)

        samples,id_list,rsid_list,col_property = None,None,None,None
        must_write_metadata2 = False

        metadata2 = self._metadata2_file_name()
        if os.path.exists(metadata2):
            d = np.load(metadata2)
            samples = d['samples']
            id_list = d['id_list'] if 'id_list' in d else None
            rsid_list = d['rsid_list'] if 'rsid_list' in d else None
            col_property = d['col_property']

        #LATER want this mesasage? logging.info("Reading and saving variant and sample metadata")
        if samples is None:
            samples =  np.array(self._read_bgen['samples'],dtype='str')
            must_write_metadata2 = True
        self._row = np.array([self._iid_function(sample) for sample in samples],dtype='str')

        if self._sid_function == 'id':
            if id_list is None:
                id_list = np.array(self._read_bgen['variants']['id'],dtype='str')
                must_write_metadata2 = True
            self._col = np.array(id_list,dtype='str')
        elif self._sid_function == 'rsid':
            if rsid_list is None:
                rsid_list = np.array(self._read_bgen['variants']['rsid'],dtype='str')
                must_write_metadata2 = True
            self._col = np.array(rsid_list,dtype='str')
        else:
            if id_list is None:
                id_list = np.array(self._read_bgen['variants']['id'],dtype='str')
                must_write_metadata2 = True
            if rsid_list is None:
                rsid_list = np.array(self._read_bgen['variants']['rsid'],dtype='str')
                must_write_metadata2 = True
            self._col = np.array([self._sid_function(id,rsid) for id,rsid in zip(id_list,rsid_list)],dtype='str')

        if len(self._col)>0: #spot check
            assert list(self._read_bgen['variants'].loc[0, "nalleles"])[0]==2, "Expect nalleles==2"

        if col_property is None:
            col_property = np.zeros((len(self._col),3),dtype='float')
            col_property[:,0] = self._read_bgen['variants']['chrom']
            col_property[:,2] = self._read_bgen['variants']['pos']
            must_write_metadata2 = True
        self._col_property = col_property

        if must_write_metadata2:
            assert id_list is not None or rsid_list is not None, "Expect either id or rsid to be used"
            if id_list is None:
                np.savez(metadata2,samples=samples,rsid_list=rsid_list,col_property=self._col_property)
            elif rsid_list is None:
                np.savez(metadata2,samples=samples,id_list=id_list,col_property=self._col_property)
            else:
                np.savez(metadata2,samples=samples,id_list=id_list,rsid_list=rsid_list,col_property=self._col_property)

        self._assert_iid_sid_pos(check_val=False)

    def _read(self, iid_index_or_none, sid_index_or_none, order, dtype, force_python_only, view_ok):
        self._run_once()

        if order=='A':
            order='F'
        dtype = np.dtype(dtype)

        iid_count_in = self.iid_count #!!!similar code elsewhere
        sid_count_in = self.sid_count

        if iid_index_or_none is not None:
            iid_count_out = len(iid_index_or_none)
            iid_index_out = iid_index_or_none
        else:
            iid_count_out = iid_count_in
            iid_index_out = None

        if sid_index_or_none is not None:
            sid_count_out = len(sid_index_or_none)
            sid_index_out = sid_index_or_none
        else:
            sid_count_out = sid_count_in
            sid_index_out = list(range(sid_count_in))

        
        val = np.zeros((iid_count_out, sid_count_out,3), order=order, dtype=dtype)

        genotype = self._read_bgen["genotype"]
        for index_out,index_in in enumerate(sid_index_out):
            g = genotype[index_in].compute()
            probs = g['probs']
            assert not g["phased"], "Expect unphased data"
            if probs.shape[0] > 0:
                assert g["ploidy"][0]==2, "Expect ploidy==2"
            assert probs.shape[-1]==3, "Expect exactly three probability values, e.g. from unphased, 2 alleles, diploid data"
            val[:,index_out,:] = (probs[iid_index_out,:] if iid_index_or_none is not None else probs)
        
        return val

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.filename)

    def __del__(self):
        self.flush()

    def flush(self):
        '''Close the \*.bgen file for reading. (If values or properties are accessed again, the file will be reopened.)

        >>> from __future__ import print_function #Python 2 & 3 compatibility
        >>> from pysnptools.distreader import Bgen
        >>> data_on_disk = Bgen('../examples/example.bgen')
        >>> print((data_on_disk.iid_count, data_on_disk.sid_count))
        (500, 199)
        >>> data_on_disk.flush()

        '''
        if hasattr(self,'_ran_once') and self._ran_once:
            self._ran_once = False
            if hasattr(self,'_read_bgen') and self._read_bgen is not None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
                del self._read_bgen
                self._read_bgen = None
            


    @staticmethod
    def write(filename, distreader, bits=16, compression=None, sample_function=default_sample_function, id_rsid_function=default_id_rsid_function, iid_function=default_iid_function, sid_function=default_sid_function, block_size=None, qctool_path=None, cleanup_temp_files=True):
        """Writes a :class:`DistReader` to BGEN format and return a the :class:`.Bgen`. Requires access to the 3rd party QCTool.

        :param filename: the name of the file to create
        :type filename: string
        :param distreader: The data that should be written to disk. It can also be any distreader, for example, :class:`.DistNpz`, :class:`.DistData`, or
           another :class:`.Bgen`.
        :type distreader: :class:`DistReader`
        :param bits: Number of bits, between 1 and 32 used to represent each 0-to-1 probability value. Default is 16.
            An np.float32 needs 23 bits. A np.float64 would need 52 bits, which the BGEN format doesn't offer, so use 32.
        :type bits: int
        :param compression: How to compress the file. Can be None (default), 'zlib', or 'zstd'.
        :type compression: bool
        :param sample_function: Function to turn a :attr:`DistReader.iid` into a BGEN sample.
           (Default: :meth:`bgen.default_sample_function`.)
        :type sample_function: function
        :param id_rsid_function: Function to turn a  a :attr:`DistReader.sid` into a BGEN (SNP) id and rsid.
           (Default: :meth:`bgen.default_id_rsid_function`.)
        :type id_rsid_function: function
        :param iid_function: Function to turn a BGEN sample into a :attr:`DistReader.iid`.
           (Default: :meth:`bgen.default_iid_function`.)
        :type iid_function: function
        :param sid_function: Function to turn a BGEN (SNP) id and rsid into a :attr:`DistReader.sid`.
           (Default: :meth:`bgen.default_sid_function`.)
        :type sid_function: function
        :param block_size: The number of SNPs to read in a batch from *distreader*. Defaults to a *block_size* such that *block_size* \* *iid_count* is about 100,000.
        :type block_size: number
        :param qctool_path: Tells the path to the 3rd party `QCTool <https://www.well.ox.ac.uk/~gav/qctool_v2/>`_. Defaults to reading
           path from environment variable QCTOOLPATH. (To use on Windows, install Ubuntu for Windows, install QCTool in Ubuntu,
           and then give the path as "ubuntu run <UBUNTU PATH TO QCTOOL".)
        :type qctool_path: string
        :param cleanup_temp_files: Tells if delete temporary \*.gen and \*.sample files.
        :type cleanup_temp_files: bool
        :rtype: :class:`.Bgen`

        >>> from pysnptools.distreader import DistHdf5, Bgen
        >>> import pysnptools.util as pstutil
        >>> distreader = DistHdf5('../examples/toydata.snpmajor.dist.hdf5')[:,:10] # A reader for the first 10 SNPs in Hdf5 format
        >>> pstutil.create_directory_if_necessary("tempdir/toydata10.bgen")
        >>> Bgen.write("tempdir/toydata10.bgen",distreader)        # Write data in BGEN format
        Bgen('tempdir/toydata10.bgen')
        """
        qctool_path = qctool_path or os.environ.get('QCTOOLPATH')
        assert qctool_path is not None, "Bgen.write() requires a path to an external qctool program either via the qctool_path input or by setting the QCTOOLPATH environment variable."

        genfile =  os.path.splitext(filename)[0]+'.gen'
        samplefile =  os.path.splitext(filename)[0]+'.sample'
        metadata =  filename+'.metadata'
        metadatanpz =  filename+'.metadata.npz'

        #We need the +1 so that all three values will have enough precision to be very near 1
        #The max(3,..) is needed to even 1 bit will have enough precision in the gen file
        decimal_places = max(3,math.ceil(math.log(2**bits,10))+1)
        Bgen.genwrite(genfile,distreader,decimal_places,id_rsid_function,sample_function,block_size)
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(metadata):
            os.remove(metadata)
        if os.path.exists(metadatanpz):
            os.remove(metadatanpz)
        cmd = '{0} -g {1} -s {2} -og {3}{4}{5}'.format(qctool_path,genfile,samplefile,filename,
                            ' -bgen-bits {0}'.format(bits) if bits is not None else '',
                            ' -bgen-compression {0}'.format(compression) if compression is not None else '')
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            print("Status : FAIL", exc.returncode, exc.output)
            raise Exception("qctool command failed")
        if cleanup_temp_files:
            os.remove(genfile)
            os.remove(samplefile)
        return Bgen(filename, iid_function=iid_function, sid_function=sid_function)

    @staticmethod
    def genwrite(filename, distreader, decimal_places=None, id_rsid_function=default_id_rsid_function, sample_function=default_sample_function, block_size=None):
        """Writes a :class:`DistReader` to Gen format

        :param filename: the name of the file to create (will also create *filename_without_gen*.sample file.)
        :type filename: string
        :param distreader: The data that should be written to disk. It can also be any distreader, for example, :class:`.DistNpz`, :class:`.DistData`, or
           another :class:`.Bgen`.
        :type distreader: :class:`DistReader`
        :param decimal_places: (Default: None) Number of decimal places with which to write the text numbers. *None* writes to full precision.
        :type bits: int or None
        :param id_rsid_function: Function to turn a  a :attr:`DistReader.sid` into a GEN (SNP) id and rsid.
           (Default: :meth:`bgen.default_id_rsid_function`.)
        :type id_rsid_function: function
        :param sid_function: Function to turn a GEN (SNP) id and rsid into a :attr:`DistReader.sid`.
           (Default: :meth:`bgen.default_sid_function`.)
        :type sid_function: function
        :param block_size: The number of SNPs to read in a batch from *distreader*. Defaults to a *block_size* such that *block_size* \* *iid_count* is about 100,000.
        :type block_size: number
        :rtype: None

        >>> from pysnptools.distreader import DistHdf5, Bgen
        >>> import pysnptools.util as pstutil
        >>> distreader = DistHdf5('../examples/toydata.snpmajor.dist.hdf5')[:,:10] # A reader for the first 10 SNPs in Hdf5 format
        >>> pstutil.create_directory_if_necessary("tempdir/toydata10.bgen")
        >>> Bgen.genwrite("tempdir/toydata10.gen",distreader)        # Write data in GEN format
        """
        #https://www.cog-genomics.org/plink2/formats#gen
        #https://web.archive.org/web/20181010160322/http://www.stats.ox.ac.uk/~marchini/software/gwas/file_format.html

        block_size = block_size or max((100*1000)//max(1,distreader.row_count),1)

        if decimal_places is None:
            format_function = lambda num:'{0}'.format(num)
        else:
            format_function = lambda num:('{0:.'+str(decimal_places)+'f}').format(num)

        start = 0
        updater_freq = max(distreader.row_count * distreader.col_count // 500,1)
        with log_in_place("sid_index ", logging.INFO) as updater:
            with open(filename+'.temp','w',newline='\n') as genfp:
                while start < distreader.sid_count:
                    distdata = distreader[:,start:start+block_size].read(view_ok=True)
                    for sid_index in range(distdata.sid_count):
                        if updater_freq>1 and (start+sid_index) % updater_freq == 0:
                            updater('{0:,} of {1:,}'.format(start+sid_index,distreader.sid_count))
                        id,rsid = id_rsid_function(distdata.sid[sid_index])
                        assert id.strip()!='','id cannot be whitespace'
                        assert rsid.strip()!='','rsid cannot be whitespace'
                        genfp.write('{0} {1} {2} {3} A G'.format(int(distdata.pos[sid_index,0]),id,rsid,int(distdata.pos[sid_index,2])))
                        for iid_index in range(distdata.iid_count):
                            prob_dist = distdata.val[iid_index,sid_index,:]
                            if not np.isnan(prob_dist).any():
                                s = ' ' + ' '.join((format_function(num) for num in prob_dist))
                                genfp.write(s)
                            else:
                                genfp.write(' 0 0 0')
                        genfp.write('\n')
                    start += distdata.sid_count
        sample_filename = os.path.splitext(filename)[0]+'.sample'
        #https://www.well.ox.ac.uk/~gav/qctool_v2/documentation/sample_file_formats.html
        with open(sample_filename,'w',newline='\n') as samplefp:
            samplefp.write('ID\n')
            samplefp.write('0\n')
            for f,i in distreader.iid:
                samplefp.write('{0}\n'.format(sample_function(f,i)))

        if os.path.exists(filename):
            os.remove(filename)
        shutil.move(filename+'.temp',filename)

    def copyinputs(self, copier):
        # doesn't need to self.run_once() because only uses original inputs
        copier.input(self.filename)
        if self._sample is not None:
            copier.input(self._sample)
        metadata = self._metadata_file_name()
        if os.path.exists(metadata):
            copier.input(metadata)
        metadata2 = self._metadata2_file_name()
        if os.path.exists(metadata2):
            copier.input(metadata2)


class TestBgen(unittest.TestCase):

    @staticmethod
    def assert_approx_equal(distdata0,distdata1,atol):
        from pysnptools.pstreader import PstData

        assert PstData._allclose(distdata0.row,distdata1.row,equal_nan=True)
        assert PstData._allclose(distdata0.col,distdata1.col,equal_nan=True)
        assert PstData._allclose(distdata0.row_property,distdata1.row_property,equal_nan=True)
        assert PstData._allclose(distdata0.col_property,distdata1.col_property,equal_nan=True)
        np.testing.assert_allclose(distdata0.val,distdata1.val,atol=atol,equal_nan=True,verbose=True)


    def test1(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        logging.info("in TestBgen test1")
        bgen = Bgen('../examples/example.bgen')
        distdata = bgen.read()
        bgen2 = bgen[:2,::3]
        distdata2 = bgen2.read()
        os.chdir(old_dir)

    def test2(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        logging.info("in TestBgen test2")
        bgen = Bgen('../examples/bits1.bgen')
        distdata = bgen.read()
        bgen2 = bgen[:2,::3]
        distdata2 = bgen2.read()
        os.chdir(old_dir)

    def test_read_write_round_trip(self):
        from pysnptools.distreader import DistGen

        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        assert 'QCTOOLPATH' in os.environ, "To run test_read_write_round_trip, QCTOOLPATH environment variable must be set. (On Windows, install QcTools in 'Ubuntu on Windows' and set to 'ubuntu run <qctoolLinuxPath>')."

        exampledata = Bgen('../examples/example.bgen')[:,10].read()
        distgen0data = DistGen(seed=332,iid_count=50,sid_count=5).read()

        for i,distdata0 in enumerate([distgen0data,exampledata]):
            for bits in list(range(1,33)):
                logging.info("input#={0},bits={1}".format(i,bits))
                file1 = 'temp/roundtrip1-{0}-{1}.bgen'.format(i,bits)
                distdata1 = Bgen.write(file1,distdata0,bits=bits,compression='zlib',cleanup_temp_files=False).read()
                distdata2 = Bgen(file1,).read()
                assert distdata1.allclose(distdata2,equal_nan=True)
                atol=1.0/(2**(bits or 16))
                if (bits or 16) == 1:
                    atol *= 1.4 # because values must add up to 1, there is more rounding error
                TestBgen.assert_approx_equal(distdata0,distdata1,atol=atol)
        os.chdir(old_dir)

    def test_bad_sum(self):
        from pysnptools.distreader import DistGen

        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        assert 'QCTOOLPATH' in os.environ, "To run test_read_write_round_trip, QCTOOLPATH environment variable must be set. (On Windows, install QcTools in 'Ubuntu on Windows' and set to 'ubuntu run <qctoolLinuxPath>')."

        distdata = Bgen('../examples/example.bgen')[:5,:5].read()

        #Just one NaN
        distdata.val[0,0,:] = [np.nan,.5,.5]
        bgen = Bgen.write('temp/should_be_all_nan.bgen',distdata)
        assert np.isnan(bgen[0,0].read().val).all()

        #Just one NaN
        distdata.val[0,0,:] = [0,0,0]
        bgen = Bgen.write('temp/should_be_all_nan2.bgen',distdata)
        assert np.isnan(bgen[0,0].read().val).all()

        #Just sums to more than 1
        distdata.val[0,0,:] = [1,2,3]
        failed = False
        try:
            bgen = Bgen.write('temp/should_fail.bgen',distdata)
        except:
            failed = True
        assert failed

        #Just sums to less than 1
        distdata.val[0,0,:] = [.2,.2,.2]
        failed = False
        try:
            bgen = Bgen.write('temp/should_fail.bgen',distdata)
        except:
            failed = True
        assert failed

        #a negative value
        distdata.val[0,0,:] = [-1,1,0]
        failed = False
        try:
            bgen = Bgen.write('temp/should_fail.bgen',distdata)
        except:
            failed = True
        assert failed
        os.chdir(old_dir)


        


    def test_read1(self):

        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))

        file_from = '../examples/example.bgen'
        file_to = 'temp/example.bgen'
        pstutil.create_directory_if_necessary(file_to)
        if os.path.exists(file_to+".metadata"):
            os.remove(file_to+".metadata")
        if os.path.exists(file_to+".metadata.npz"):
            os.remove(file_to+".metadata.npz")
        shutil.copy(file_from,file_to)

        for loop_index in range(2):
            iid_function = lambda bgen_sample_id: (bgen_sample_id,bgen_sample_id) #Use the bgen_sample_id for both parts of iid
            bgen = Bgen(file_to)
            assert np.array_equal(bgen.iid[0],['0', 'sample_001'])
            assert bgen.sid[0]=='SNPID_2,RSID_2'
            bgen = Bgen(file_to,iid_function,sid_function='id')
            assert np.array_equal(bgen.iid[0],['sample_001', 'sample_001'])
            assert bgen.sid[0]=='SNPID_2'
            bgen = Bgen(file_to,iid_function,sid_function='rsid')
            assert np.array_equal(bgen.iid[0],['sample_001', 'sample_001'])
            assert bgen.sid[0]=='RSID_2'
            sid_function = lambda id,rsid: '{0},{1}'.format(id,rsid)
            bgen = Bgen(file_to,iid_function,sid_function=sid_function)
            assert bgen.sid[0]=='SNPID_2,RSID_2'

        os.remove(file_to+".metadata.npz")
        sid_function = lambda id,rsid: '{0},{1}'.format(id,rsid)
        bgen = Bgen(file_to,iid_function,sid_function=sid_function)
        assert bgen.sid[0]=='SNPID_2,RSID_2'

        os.remove(file_to+".metadata.npz")
        bgen = Bgen(file_to,iid_function,sid_function='rsid')
        assert np.array_equal(bgen.iid[0],['sample_001', 'sample_001'])
        assert bgen.sid[0]=='RSID_2'

        os.chdir(old_dir)



        #This and many of the tests based on bgen-reader-py\bgen_reader\test
    def test_bgen_samples_inside_bgen(self):
        with example_files("haplotypes.bgen") as filepath:
            data = Bgen(filepath)
            samples = [("0","sample_0"), ("0","sample_1"), ("0","sample_2"), ("0","sample_3")]
            assert (data.iid == samples).all()


    def test_bgen_samples_not_present(self):
        with example_files("complex.23bits.no.samples.bgen") as filepath:
            data = Bgen(filepath)
            samples = [("0","sample_0"), ("0","sample_1"), ("0","sample_2"), ("0","sample_3")]
            assert (data.iid == samples).all()


    def test_bgen_samples_specify_samples_file(self):
        with example_files(["complex.23bits.bgen", "complex.sample"]) as filepaths:
            data = Bgen(filepaths[0], sample=filepaths[1], verbose=False)
            assert (data.iid[:,1] == ["sample_0", "sample_1", "sample_2", "sample_3"]).all()

    def test_metafile_provided(self):
        filenames = ["haplotypes.bgen", "haplotypes.bgen.metadata.valid"]
        with example_files(filenames) as filepaths:
            bgen = Bgen(filepaths[0], metadata=filepaths[1], verbose=False)
            bgen.iid



    def test_bgen_reader_variants_info(self):
        with example_files("example.32bits.bgen") as filepath:
            bgen = Bgen(filepath,sid_function='id')

            assert(bgen.pos[0,0]==1)
            assert(bgen.sid[0]=="SNPID_2")
            assert(bgen.pos[0,2] == 2000)

            assert(bgen.pos[7,0]==1)
            assert(bgen.sid[7]=="SNPID_9")
            assert(bgen.pos[7,2] == 9000)

            assert(bgen.pos[-1,0]==1)
            assert(bgen.sid[-1]=="SNPID_200")
            assert(bgen.pos[-1,2] == 100001)

            assert((bgen.iid[0] == ("0", "sample_001")).all())
            assert((bgen.iid[7] == ("0", "sample_008")).all())
            assert((bgen.iid[-1] == ("0", "sample_500")).all())

            g = bgen[0,0].read()
            assert(np.isnan(g.val).all())

            g = bgen[1,0].read()
            a = [[[0.027802362811705648, 0.00863673794284387, 0.9635608992454505]]]
            np.testing.assert_array_almost_equal(g.val, a)

            b = [[[
                0.97970582847010945215516,
                0.01947019668749305418287,
                0.00082397484239749366197,
            ]]]
            g = bgen[2,1].read()
            np.testing.assert_array_almost_equal(g.val, b)


    

    def test_bgen_reader_without_metadata(self):
        with example_files("example.32bits.bgen") as filepath:
            bgen = Bgen(filepath)
            variants = bgen.read()
            samples = bgen.iid
            assert samples[-1,1]=="sample_500"


    def test_bgen_reader_file_notfound(self):
            bgen = Bgen("/1/2/3/example.32bits.bgen")
            try:
                bgen.iid # expect error
                got_error = False
            except Exception as e:
                got_error = True
            assert got_error


    def test_create_metadata_file(self):
        with example_files("example.32bits.bgen") as filepath:
            folder = os.path.dirname(filepath)
            metafile_filepath = os.path.join(folder, filepath + ".metadata")
            
            try:
                bgen = Bgen(filepath)
                bgen.iid
                assert(os.path.exists(metafile_filepath))
            finally:
                if os.path.exists(metafile_filepath):
                    os.remove(metafile_filepath)

    def test_doctest(self):
        import pysnptools.distreader.bgen
        import doctest

        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        result = doctest.testmod(pysnptools.distreader.bgen)
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__

    


def getTestSuite():
    """
    set up composite test suite
    """
    
    test_suite = unittest.TestSuite([])
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestBgen))
    return test_suite



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    if False:
        from pysnptools.distreader import Bgen
        bgen = Bgen(r'M:\deldir\2500x100.bgen',verbose=True)
        bgen.read()
        print(bgen.shape)
        print("")

    if False:
        from pysnptools.distreader import Bgen
        bgen = Bgen(r'M:\deldir\1x1000000.bgen',verbose=True)
        print(bgen.shape)
        print("")


    if False:
        from pysnptools.distreader import Bgen
        bgen2 = Bgen(r'M:\deldir\10x5000000.bgen',verbose=True)
        print(bgen2.shape)

    if False: 
        #iid_count = 500*1000
        #sid_count = 100
        #iid_count = 1
        #sid_count = 1*1000*1000
        iid_count = 2500
        sid_count = 100

        from pysnptools.distreader import DistGen
        from pysnptools.distreader import Bgen
        distgen = DistGen(seed=332,iid_count=iid_count,sid_count=sid_count)
        Bgen.write('{0}x{1}.bgen'.format(iid_count,sid_count),distgen)
    if False:
        from pysnptools.distreader import Bgen
        bgen = Bgen(r'M:\deldir\10x5000000.bgen')
        print(bgen.iid)


    import doctest
    result = doctest.testmod()
    assert result.failed == 0, "failed doc test: " + __file__

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True)
    r.run(suites)

