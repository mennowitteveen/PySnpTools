import os
import logging
import numpy as np
import warnings
import unittest
from bgen_reader import read_bgen
from bgen_reader import example_files
from bgen_reader import create_metafile
from pysnptools.util import log_in_place
import shutil
import math
from tempfile import TemporaryFile
import subprocess
import pysnptools.util as pstutil
from pysnptools.distreader import DistReader

#!!!cmk document these
def default_iid_function(sample):
    fields = sample.split(',')
    if len(fields)==2:
        return fields[0],fields[1]
    else:
        return ('0',sample)

def default_sid_function(id,rsid):
    if rsid=='0' or rsid=='':
        return id
    else:
        return id+','+rsid

def default_sample_function(famid,indid):
    if famid=='0' or famid=='':
        return indid
    else:
        return famid+','+indid

def default_id_rsid_function(sid):
    fields = sid.split(',')
    if len(fields)==2:
        return fields[0],fields[1]
    else:
        return sid,'0'



class Bgen(DistReader):
    '''
    A :class:`.DistReader` for reading \*.dist.hdf5 files from disk.#!!!cmk update

    See :class:`.DistReader` for general examples of using DistReaders.

    The general HDF5 format is described in http://www.hdfgroup.org/HDF5/. The DistHdf5 format stores
    val, iid, sid, and pos information in Hdf5 format.
   
    **Constructor:**
        :Parameters: * **filename** (*string*) -- The DistHdf5 file to read.

        :Example: cmk update doc

        >>> from __future__ import print_function #Python 2 & 3 compatibility
        >>> from pysnptools.distreader import DistHdf5
        >>> data_on_disk = DistHdf5('../examples/toydata.snpmajor.dist.hdf5')
        >>> print((data_on_disk.iid_count, data_on_disk.sid_count))
        (25, 10000)

    **Methods beyond** :class:`.DistReader`

    '''
    warning_dictionary = {}
    def __init__(self, filename, iid_function=default_iid_function, sid_function=default_sid_function, verbose=False, metadata=None, sample=None):
        #!!!cmk document that sid_function can be 'id' or 'rsid' and will be faster
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

        #Warn about reopening files with new contents #!!!cmk remove when bug is fixed
        new_file_date = os.stat(self.filename).st_ctime
        old_file_date = Bgen.warning_dictionary.get(self.filename)
        if old_file_date is not None and old_file_date != new_file_date:
            logging.warning('Opening a file again, but its creation date has changed See https://github.com/limix/bgen-reader-py/issues/25. File "{0}"'.format(self.filename))
        else:
            Bgen.warning_dictionary[self.filename] = new_file_date

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

        #!!!cmk want this mesasage? logging.info("Reading and saving variant and sample metadata")
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
            assert list(self._read_bgen['variants'].loc[0, "nalleles"])[0]==2, "Expect nalleles==2" #!!!cmk test that this is fast even with 1M sid_count

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
        '''Flush :attr:`.DistMemMap.val` to disk and close the file. (If values or properties are accessed again, the file will be reopened.)#!!!cmk update doc
        cmk update doc

        >>> import pysnptools.util as pstutil
        >>> from pysnptools.distreader import DistMemMap
        >>> filename = "tempdir/tiny.dist.memmap"
        >>> pstutil.create_directory_if_necessary(filename)
        >>> dist_mem_map = DistMemMap.empty(iid=[['fam0','iid0'],['fam0','iid1']], sid=['snp334','snp349','snp921'],filename=filename,order="F",dtype=np.float64)
        >>> dist_mem_map.val[:,:,:] = [[[.5,.5,0],[0,0,1],[.5,.5,0]],
        ...                            [[0,1.,0],[0,.75,.25],[.5,.5,0]]]
        >>> dist_mem_map.flush()

        '''
        if hasattr(self,'_ran_once') and self._ran_once:
            self._ran_once = False
            if hasattr(self,'_read_bgen') and self._read_bgen is not None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
                del self._read_bgen
                self._read_bgen = None
            


    @staticmethod
    def write(filename, distreader, bits=None, compression=None, sample_function=default_sample_function, id_rsid_function=default_id_rsid_function, iid_function=default_iid_function, sid_function=default_sid_function, sid_batch_size=None, qctool_path=None, cleanup_temp_files=True):
        """
        cmk update doc

        """
        #!!!cmk doc WARN that 32 bits here, the max, corresponds to 10 decial places of precession and needs a dtype of float64 to capture.
        #!!!cmk a dtype of float32 correponds to 23 bgen bits (7 decimal places)
        #cmk doc: nan, 3 0's, one nan will all turn into 3 nan's. Negative, sum not 1 will raise error
        #cmk doc that compression can be blank or zlib or zstd 
        #cmk doc that default bits seems to be 16


        qctool_path = qctool_path or os.environ.get('QCTOOLPATH')
        assert qctool_path is not None, "Bgen.write() requires a path to an external qctool program either via the qctool_path input or by setting the QCTOOLPATH environment variable."

        genfile =  os.path.splitext(filename)[0]+'.gen'
        samplefile =  os.path.splitext(filename)[0]+'.sample'
        metadata =  filename+'.metadata'
        metadatanpz =  filename+'.metadata.npz'

        bits = bits or 16
        #We need the +1 so that all three values will have enough precision to be very near 1
        #The max(3,..) is needed to even 1 bit will have enough precision in the gen file
        decimal_places = max(3,math.ceil(math.log(2**bits,10))+1)
        Bgen.genwrite(genfile,distreader,decimal_places,id_rsid_function,sample_function,sid_batch_size)
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
    def genwrite(filename, distreader, decimal_places=None, id_rsid_function=default_id_rsid_function, sample_function=default_sample_function, sid_batch_size=None):
        """Writes a :class:`DistReader` to Gen format and returns None #!!!cmk update docs

        :param filename: the name of the file to create
        :type filename: string
        :param distdata: The in-memory data that should be written to disk.
        :type distdata: :class:`DistData`
        :rtype: :class:`.DistNpz`

        cmk update doc

        >>> from pysnptools.distreader import DistNpz, DistHdf5
        >>> import pysnptools.util as pstutil
        >>> distdata = DistHdf5('../examples/toydata.iidmajor.dist.hdf5')[:,:10].read()     # Read first 10 snps from DistHdf5 format
        >>> pstutil.create_directory_if_necessary("tempdir/toydata10.dist.npz")
        >>> DistNpz.write("tempdir/toydata10.dist.npz",distdata)          # Write data in DistNpz format
        DistNpz('tempdir/toydata10.dist.npz')
        """
        #https://www.cog-genomics.org/plink2/formats#gen
        #https://web.archive.org/web/20181010160322/http://www.stats.ox.ac.uk/~marchini/software/gwas/file_format.html

        sid_batch_size = sid_batch_size or max((100*1000)//max(1,distreader.row_count),1)

        if decimal_places is None:
            format_function = lambda num:'{0}'.format(num)
        else:
            format_function = lambda num:('{0:.'+str(decimal_places)+'f}').format(num)

        start = 0
        updater_freq = max(distreader.row_count * distreader.col_count // 500,1)
        with log_in_place("sid_index ", logging.INFO) as updater:
            with open(filename+'.temp','w',newline='\n') as genfp:
                while start < distreader.sid_count:
                    distdata = distreader[:,start:start+sid_batch_size].read(view_ok=True)
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
            for bits in [None]+list(range(1,33)):
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

    #!!!cmk

    if False:
        from pysnptools.distreader import Bgen
        bgen = Bgen(r'M:\deldir\2500x100.bgen',verbose=True)
        bgen.read()
        print(bgen.shape)
        print("cmk")

    if False:
        from pysnptools.distreader import Bgen
        bgen = Bgen(r'M:\deldir\1x1000000.bgen',verbose=True)
        print(bgen.shape)
        print("cmk")


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


    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True)
    r.run(suites)

    import doctest
    result = doctest.testmod()
    assert result.failed == 0, "failed doc test: " + __file__
