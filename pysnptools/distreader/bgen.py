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


#iid_function
def _fill_famid_with_0(sample):
    return ('0',sample)

#sample_function
def _ignore_famid(iid_value):
    return iid_value[1]

#id_rsid_function
def _sid_to_id_with_no_rsid(sid_value):
    return (sid_value,'0')



class Bgen(DistReader):
    '''
    A :class:`.DistReader` for reading \*.dist.hdf5 files from disk.#!!!cmk update

    See :class:`.DistReader` for general examples of using DistReaders.

    The general HDF5 format is described in http://www.hdfgroup.org/HDF5/. The DistHdf5 format stores
    val, iid, sid, and pos information in Hdf5 format.
   
    **Constructor:**
        :Parameters: * **filename** (*string*) -- The DistHdf5 file to read.

        :Example:

        >>> from __future__ import print_function #Python 2 & 3 compatibility
        >>> from pysnptools.distreader import DistHdf5
        >>> data_on_disk = DistHdf5('../examples/toydata.snpmajor.dist.hdf5')
        >>> print((data_on_disk.iid_count, data_on_disk.sid_count))
        (25, 10000)

    **Methods beyond** :class:`.DistReader`

    '''
    def __init__(self, filename, verbose=False, iid_function=None, sid_function='id'):
        super(Bgen, self).__init__()
        self._ran_once = False
        self.read_bgen = None
        iid_function = iid_function or _fill_famid_with_0

        self.filename = filename
        self._verbose = verbose
        self._iid_function = iid_function
        self._sid_function = sid_function

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

        self._read_bgen = read_bgen(self.filename,verbose=self._verbose)

        metadata2 = self.filename + ".metadata.npz"
        samples,id_list,rsid_list,col_property = None,None,None,None
        must_write_metadata2 = False
        if os.path.exists(metadata2):
            d = np.load(metadata2)
            samples = d['samples']
            id_list = d['id_list'] if 'id_list' in d else None
            rsid_list = d['rsid_list'] if 'rsid_list' in d else None
            col_property = d['col_property']

        #!!!cmk23 want this mesasage? logging.info("Reading and saving variant and sample metadata")
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
            self._col = np.array([self._sid_function(id,rsid) for id,rsid in zip(id_list,rsid_list)],dtype='str') #!!!cmk23 test this

        if len(self._col)>0: #spot check
            assert list(self._read_bgen['variants'].loc[0, "nalleles"])[0]==2, "Expect nalleles==2" #!!!cmk23 test that this is fast even with 1M sid_count

        if col_property is None:
            col_property = np.zeros((len(self._col),3),dtype='float')
            col_property[:,0] = self._read_bgen['variants']['chrom'] #!!!cmk if this doesn't parse as numbers, leave it a zeros
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

        iid_count_in = self.iid_count #!!!cmk similar code elsewhere
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

    def __del__(self):#!!!cmk
        self.flush()

    #!!!cmk23 test it
    def flush(self):
        '''Flush :attr:`.DistMemMap.val` to disk and close the file. (If values or properties are accessed again, the file will be reopened.)#!!!cmk update doc

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
                del self._read_bgen #!!!cmk is this needed? Meaningful?
                self._read_bgen = None
            


    #!!!cmk confirm that na na na 0,0,0 round trips
    #!!!cmk write a test for this
    #!!!cmk22 name these snpid_function, etc
    #!!!cmk22 can we set all batch sizes on this file automatically better?
    @staticmethod
    def write(filename, distreader, bits=None, compression=None, sample_function=None, id_rsid_function=None, iid_function=None, sid_function='id', sid_batch_size=100, qctool_path=None, cleanup_temp_files=True):
        ""#!!!cmk doc WARN that 32 bits here, the max, corresponds to 10 decial places of precession and needs a dtype of float64 to capture.
         #!!!cmk a dtype of float32 correponds to 23 bgen bits (7 decimal places)
         #!!!cmk23 what happens if you try to write out something too big to be a probabliy, nor negative, or adds up to more than 1?
         #compression can be blank or zlib or zstd 
         #default bits seems to be 16
        iid_function=iid_function or _fill_famid_with_0

        if qctool_path is None:
            key = 'QCTOOLPATH'
            qctool_path = os.environ.get(key)
            assert qctool_path is not None, "Bgen.write() requires a path to an external qctool program either via the qctool_path input or by setting the QCTOOLPATH environment variable."
        genfile =  os.path.splitext(filename)[0]+'.gen'
        samplefile =  os.path.splitext(filename)[0]+'.sample'
        metadata =  filename+'.metadata'
        metadatanpz =  filename+'.metadata.npz'
        if bits is None:
            decimal_places=None
        else:
             #We need the +1 so that all three values will have enough precision to be very near 1
             #The max(3,..) is needed to even 1 bit will have enough precision in the gen file
            decimal_places = max(3,math.ceil(math.log(2**bits,10))*2) #!!!cmk change +2 to +1
        Bgen.genwrite(genfile,distreader,decimal_places,id_rsid_function,sample_function,sid_batch_size)
        cmd = '{0} -g {1} -s {2} -og {3}{4}{5}'.format(qctool_path,genfile,samplefile,filename,
                            ' -bgen-bits {0}'.format(bits) if bits is not None else '',
                            ' -bgen-compression {0}'.format(compression) if compression is not None else '')
        if os.path.exists(metadata):
            os.remove(metadata)
        if os.path.exists(metadatanpz):
            os.remove(metadatanpz)
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True, universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            print("Status : FAIL", exc.returncode, exc.output)
            raise Exception("qctool command failed")
        if cleanup_temp_files:
            os.remove(genfile)
            os.remove(samplefile)
        return Bgen(filename,iid_function=iid_function, sid_function=sid_function)
        #!!!cmkreturn cmd


    #!!!cmk confirm that na na na 0,0,0 round trips
    #!!!cmk write a test for this
    @staticmethod
    def genwrite(filename, distreader, decimal_places=None, id_rsid_function=None, sample_function=None, sid_batch_size=100):
        """Writes a :class:`DistReader` to Gen format and returns None #!!!cmk update docs

        :param filename: the name of the file to create
        :type filename: string
        :param distdata: The in-memory data that should be written to disk.
        :type distdata: :class:`DistData`
        :rtype: :class:`.DistNpz`

        >>> from pysnptools.distreader import DistNpz, DistHdf5
        >>> import pysnptools.util as pstutil
        >>> distdata = DistHdf5('../examples/toydata.iidmajor.dist.hdf5')[:,:10].read()     # Read first 10 snps from DistHdf5 format
        >>> pstutil.create_directory_if_necessary("tempdir/toydata10.dist.npz")
        >>> DistNpz.write("tempdir/toydata10.dist.npz",distdata)          # Write data in DistNpz format
        DistNpz('tempdir/toydata10.dist.npz')
        """
        #https://www.cog-genomics.org/plink2/formats#gen
        #https://web.archive.org/web/20181010160322/http://www.stats.ox.ac.uk/~marchini/software/gwas/file_format.html

        id_rsid_function = id_rsid_function or _sid_to_id_with_no_rsid
        sample_function = sample_function  or _ignore_famid

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
                        if (start+sid_index) % updater_freq == 0: #!!!cmk23 test this
                            updater('{0:,} of {1:,}'.format(start+sid_index,distreader.sid_count))#!!!cmk23 print commas in numbers test this
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

class TestBgen(unittest.TestCase):    #!!!cmk23 be sure these are run

    @staticmethod
    def assert_approx_equal(distdata0,distdata1,atol):
        from pysnptools.pstreader import PstData

        assert PstData._allclose(distdata0.row,distdata1.row,equal_nan=True)
        assert PstData._allclose(distdata0.col,distdata1.col,equal_nan=True)
        assert PstData._allclose(distdata0.row_property,distdata1.row_property,equal_nan=True)
        assert PstData._allclose(distdata0.col_property,distdata1.col_property,equal_nan=True)
        np.testing.assert_allclose(distdata0.val,distdata1.val,atol=atol,equal_nan=True,verbose=True)


    def cmk22test1(self):
        logging.info("in TestBgen test1")
        bgen = Bgen('../examples/example.bgen')
        #print(bgen.iid,bgen.sid,bgen.pos)
        distdata = bgen.read()
        #print(distdata.val)
        bgen2 = bgen[:2,::3]
        #print(bgen2.iid,bgen2.sid,bgen2.pos)
        distdata2 = bgen2.read()
        #print(distdata2.val)

    def cmk22test2(self):
        logging.info("in TestBgen test2")
        bgen = Bgen('../examples/bits1.bgen')
        #print(bgen.iid,bgen.sid,bgen.pos)
        distdata = bgen.read()
        #print(distdata.val)
        bgen2 = bgen[:2,::3]
        #print(bgen2.iid,bgen2.sid,bgen2.pos)
        distdata2 = bgen2.read()
        #print(distdata2.val)

    #!!!cmk23 make these the default
    @staticmethod
    def iid_function(sample):
        return ('0',sample)
    @staticmethod
    def sid_function(id,rsid):
        if rsid=='0':
            return id
        else:
            return id+','+rsid
    @staticmethod
    def id_rsid_function(sid):
        fields = sid.split(',')
        if len(fields)==2:
            return fields[0],fields[1]
        else:
            return sid,'0'
    def test_abs_error(self):
        file0 = '../examples/example.bgen'
        example = Bgen(file0,iid_function=TestBgen.iid_function,sid_function=TestBgen.sid_function)
        bgen0 = example[:,20]
        distdata0 = bgen0.read()
        file1 = 'temp/abs_error.bgen'
        for bits in [32]+list(range(1,32)):#!!!cmk[None]+:
            bgen1 = Bgen.write(file1,bgen0,bits=bits,compression='zlib',
                        sample_function=lambda f,i:i,
                        id_rsid_function=TestBgen.id_rsid_function,
                        iid_function=TestBgen.iid_function,
                        sid_function=TestBgen.sid_function,
                        cleanup_temp_files=False)
            distdata1=bgen1.read()
            bgen1.flush()
            abs_error = np.abs(distdata0.val-distdata1.val).max()
            print((bits,abs_error))


    def cmk22test_read_write_round_trip(self):
        from pysnptools.distreader import DistGen
        #!!!cmk23
        #if os.name == 'nt':
        #    logging.info('test_read_write_round_trip only runs on Linux')
        assert 'QCTOOLPATH' in os.environ, "To run test_read_write_round_trip, QCTOOLPATH environment variable must be set"

        file0 = '../examples/example.bgen'


        #!!!cmk why does write file when bits is set to 5?
        example = Bgen(file0,iid_function=TestBgen.iid_function,sid_function=TestBgen.sid_function)
        iid_count = 10
        sid_count = 100
        distgen0 = DistGen(seed=332,iid_count=iid_count,sid_count=sid_count)

        file1 = 'temp/roundtrip1.bgen' #!!!cmk23 what if we write a new bgen file and there is an old metadata and or metadata.npz file. At least, need to remove them
        for bgen0 in [example[:,20],distgen0]:#!!!cmk[example[:,20],distgen0]:#,example]:#!!!cmk why does [example[:,20],distgen0] fail with bit=18 or 5
            distdata0 = bgen0.read()#!!!cmk24 move out of loop (carefully)
            for bits in [20]:#!!!cmk[None]+list(range(1,32)): #!!!cmk22 why do we need to +2 instead of +1 to get error in tolerance
                logging.info("bits={0}".format(bits))
                distdata1 = Bgen.write(file1,bgen0,bits=bits,compression='zlib',
                           sample_function=lambda f,i:i,
                           id_rsid_function=TestBgen.id_rsid_function,
                           iid_function=TestBgen.iid_function,
                           sid_function=TestBgen.sid_function,
                           cleanup_temp_files=False
                           ).read()
                distdata2 = Bgen(file1,iid_function=TestBgen.iid_function,sid_function=TestBgen.sid_function).read()
                assert distdata1.allclose(distdata2,equal_nan=True)
                atol=1.0/(2**(bits or 16))
                if (bits or 16) == 1:
                    atol = 1.0 # because values must add up to 1, there is more rounding error
                elif (bits or 16) == 1:
                    atol = atol*1.3
                else:
                    atol = atol*1.3
                TestBgen.assert_approx_equal(distdata0,distdata1,atol=atol)


    #    for sid_function, anti_sid_function in [[None,None],
    #                                            ['id',lambda sid:(sid,'')],
    #                                            ['rsid',lambda sid:('',sid)],
    #                                            [lambda s,r: s+','+r],lambda sid,','.split(sid)]
    #                                            ]:
                
    #        bgen0 = Bgen('../examples/example.bgen',sid_function=sid_function)
        


    def cmk22test_read1(self): #!!!cmk22 make round trip (linux) read and write exercising all the _functions.
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
            sid_function = 'id'
            bgen = Bgen(file_to,iid_function)
            assert np.array_equal(bgen.iid[0],['0', 'sample_001'])
            assert bgen.sid[0]=='SNPID_2'
            bgen = Bgen(file_to,iid_function,sid_function='id')
            assert np.array_equal(bgen.iid[0],['0', 'sample_001'])
            assert bgen.sid[0]=='SNPID_2'
            bgen = Bgen(file_to,iid_function,sid_function='rsid')
            assert np.array_equal(bgen.iid[0],['0', 'sample_001'])
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
        assert np.array_equal(bgen.iid[0],['0', 'sample_001'])
        assert bgen.sid[0]=='RSID_2'


        #This and many of the tests based on bgen-reader-py\bgen_reader\test
    def cmk22test_bgen_samples_inside_bgen(self):
        with example_files("haplotypes.bgen") as filepath:
            data = Bgen(filepath)
            samples = [("0","sample_0"), ("0","sample_1"), ("0","sample_2"), ("0","sample_3")]
            assert (data.iid == samples).all()


    def cmk22test_bgen_samples_not_present(self):
        with example_files("complex.23bits.no.samples.bgen") as filepath:
            data = Bgen(filepath)
            samples = [("0","sample_0"), ("0","sample_1"), ("0","sample_2"), ("0","sample_3")]
            assert (data.iid == samples).all()


    def cmktest_bgen_samples_specify_samples_file(self):
        with example_files(["complex.23bits.bgen", "complex.sample"]) as filepaths: #!!!cmk what's going on with *.sample????
            data = read_bgen(filepaths[0], samples_filepath=filepaths[1], verbose=False)
            samples = ["sample_0", "sample_1", "sample_2", "sample_3"]
            samples = Series(samples, dtype=str, name="id")
            assert_(all(data["samples"] == samples))

    def cmktest_metafile_provided(self):
        filenames = ["haplotypes.bgen", "haplotypes.bgen.metadata.valid"]#!!!cmk what's going on with *.sample????
        with example_files(filenames) as filepaths:
            read_bgen(filepaths[0], metafile_filepath=filepaths[1], verbose=False)


    #We don't support phased
    #def cmk22test_bgen_reader_phased_genotype(self): #!!!cmk think about support for phased
    #    with example_files("haplotypes.bgen") as filepath:
    #        bgen = Bgen(filepath, verbose=False)
    #        assert(bgen.pos[0,0] == 1)
    #        assert(bgen.sid[0] == "SNP1")
    #        assert(bgen.pos[0,2]== 1)

    #        assert(bgen.pos[2,0] == 1)
    #        assert(bgen.sid[2] == "SNP3")
    #        assert(bgen.pos[2,2]== 3)

    #        assert((bgen.iid[0] ==("","sample_0")).all())
    #        assert((bgen.iid[2] ==("","sample_2")).all())

    #        assert((bgen.iid[-1] ==("","sample_3")).all())

    #        g = bgen[0,0].read()
    #        assert_allclose(g.val, [[[1.0, 0.0, 1.0, 0.0]]]) #cmk code doesn't know about phased
    #        g = bgen[-1,-1].read()
    #        assert_allclose(g.val, [[[1.0, 0.0, 0.0, 1.0]]])



    def cmk22test_bgen_reader_variants_info(self):
        with example_files("example.32bits.bgen") as filepath:
            bgen = Bgen(filepath)

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


    def cmk_test_bgen_reader_phased_genotype(self): #!!!cmk don't know about phased
        with example_files("haplotypes.bgen") as filepath:
            bgen = read_bgen(filepath, verbose=False)
            variants = bgen["variants"].compute()
            samples = bgen["samples"]
            assert_("genotype" in bgen)

            assert_equal(variants.loc[0, "chrom"], "1")
            assert_equal(variants.loc[0, "id"], "SNP1")
            assert_equal(variants.loc[0, "nalleles"], 2)
            assert_equal(variants.loc[0, "allele_ids"], "A,G")
            assert_equal(variants.loc[0, "pos"], 1)
            assert_equal(variants.loc[0, "rsid"], "RS1")

            assert_equal(variants.loc[2, "chrom"], "1")
            assert_equal(variants.loc[2, "id"], "SNP3")
            assert_equal(variants.loc[2, "nalleles"], 2)
            assert_equal(variants.loc[2, "allele_ids"], "A,G")
            assert_equal(variants.loc[2, "pos"], 3)
            assert_equal(variants.loc[2, "rsid"], "RS3")

            assert_equal(samples.loc[0], "sample_0")
            assert_equal(samples.loc[2], "sample_2")

            n = samples.shape[0]
            assert_equal(samples.loc[n - 1], "sample_3")

            g = bgen["genotype"][0].compute()["probs"]
            a = [1.0, 0.0, 1.0, 0.0]
            assert_allclose(g[0, :], a)

            k = len(variants)
            n = len(samples)
            a = [1.0, 0.0, 0.0, 1.0]
            g = bgen["genotype"][k - 1].compute()["probs"]
            assert_allclose(g[n - 1, :], a)


    def cmktest_bgen_reader_without_metadata(self):
        with example_files("example.32bits.bgen") as filepath:
            bgen = read_bgen(filepath)
            variants = bgen["variants"].compute()
            samples = bgen["samples"]
            assert_("genotype" in bgen)
            assert_equal(variants.loc[7, "allele_ids"], "A,G")
            n = samples.shape[0]
            assert_equal(samples.loc[n - 1], "sample_500")


    #def cmk22test_bgen_reader_with_wrong_metadata_file(self):
    #    with example_files(["example.32bits.bgen", "wrong.metadata"]) as filepaths:
    #        bgen = Bgen(filepaths[0], metadata=filepaths[1])
    #        try:
    #            bgen.iid # expect error
    #            got_error = False
    #        except Exception as e:
    #            got_error = True
    #        assert got_error


    #def cmktest_bgen_reader_with_nonexistent_metadata_file(self):
    #    with example_files("example.32bits.bgen") as filepath:
    #        folder = os.path.dirname(filepath)
    #        metafile_filepath = os.path.join(folder, "nonexistent.metadata")

    #        bgen = Bgen(filepath,metadata=metafile_filepath)
    #        bgen.iid
    #        assert os.path.exists(metafile_filepath)


    def cmk22test_bgen_reader_file_notfound(self):
            bgen = Bgen("/1/2/3/example.32bits.bgen")
            try:
                bgen.iid # expect error
                got_error = False
            except Exception as e:
                got_error = True
            assert got_error


    def cmk22test_create_metadata_file(self):
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


    def cmktest_bgen_reader_complex(self):
        with example_files("complex.23bits.bgen") as filepath:
            bgen = read_bgen(filepath, verbose=False)
            variants = bgen["variants"].compute()
            samples = bgen["samples"]
            assert_("genotype" in bgen)

            assert_equal(variants.loc[0, "chrom"], "01")
            assert_equal(variants.loc[0, "id"], "")
            assert_equal(variants.loc[0, "nalleles"], 2)
            assert_equal(variants.loc[0, "allele_ids"], "A,G")
            assert_equal(variants.loc[0, "pos"], 1)
            assert_equal(variants.loc[0, "rsid"], "V1")

            assert_equal(variants.loc[7, "chrom"], "01")
            assert_equal(variants.loc[7, "id"], "")
            assert_equal(variants.loc[7, "nalleles"], 7)
            assert_equal(variants.loc[7, "allele_ids"], "A,G,GT,GTT,GTTT,GTTTT,GTTTTT")
            assert_equal(variants.loc[7, "pos"], 8)
            assert_equal(variants.loc[7, "rsid"], "M8")

            n = variants.shape[0]
            assert_equal(variants.loc[n - 1, "chrom"], "01")
            assert_equal(variants.loc[n - 1, "id"], "")
            assert_equal(variants.loc[n - 1, "nalleles"], 2)
            assert_equal(variants.loc[n - 1, "allele_ids"], "A,G")
            assert_equal(variants.loc[n - 1, "pos"], 10)
            assert_equal(variants.loc[n - 1, "rsid"], "M10")

            assert_equal(samples.loc[0], "sample_0")
            assert_equal(samples.loc[3], "sample_3")

            g = bgen["genotype"][0].compute()["probs"][0]
            assert_allclose(g[:2], [1, 0])
            assert_(isnan(g[2]))

            g = bgen["genotype"][0].compute()["probs"][1]
            assert_allclose(g[:3], [1, 0, 0])

            g = bgen["genotype"][-1].compute()["probs"][-1]
            assert_allclose(g[:5], [0, 0, 0, 1, 0])

            ploidy = bgen["genotype"][0].compute()["ploidy"]
            assert_allclose(ploidy, [1, 2, 2, 2])
            ploidy = bgen["genotype"][-1].compute()["ploidy"]
            assert_allclose(ploidy, [4, 4, 4, 4])

            nvariants = len(variants)
            phased = [bgen["genotype"][i].compute()["phased"] for i in range(nvariants)]
            phased = array(phased)
            assert_equal(phased.dtype.name, "bool")
            ideal = array([False, True, True, False, True, True, True, True, False, False])
            assert_(array_equal(phased, ideal))


    def cmktest_bgen_reader_complex_sample_file(self):
        with example_files(["complex.23bits.bgen", "complex.sample"]) as filepaths:
            bgen = read_bgen(filepaths[0], samples_filepath=filepaths[1], verbose=False)
            variants = bgen["variants"].compute()
            samples = bgen["samples"]
            assert_("genotype" in bgen)

            assert_equal(variants.loc[0, "chrom"], "01")
            assert_equal(variants.loc[0, "id"], "")
            assert_equal(variants.loc[0, "nalleles"], 2)
            assert_equal(variants.loc[0, "allele_ids"], "A,G")
            assert_equal(variants.loc[0, "pos"], 1)
            assert_equal(variants.loc[0, "rsid"], "V1")

            assert_equal(variants.loc[7, "chrom"], "01")
            assert_equal(variants.loc[7, "id"], "")
            assert_equal(variants.loc[7, "nalleles"], 7)
            assert_equal(variants.loc[7, "allele_ids"], "A,G,GT,GTT,GTTT,GTTTT,GTTTTT")
            assert_equal(variants.loc[7, "pos"], 8)
            assert_equal(variants.loc[7, "rsid"], "M8")

            n = variants.shape[0]
            assert_equal(variants.loc[n - 1, "chrom"], "01")
            assert_equal(variants.loc[n - 1, "id"], "")
            assert_equal(variants.loc[n - 1, "nalleles"], 2)
            assert_equal(variants.loc[n - 1, "allele_ids"], "A,G")
            assert_equal(variants.loc[n - 1, "pos"], 10)
            assert_equal(variants.loc[n - 1, "rsid"], "M10")

            assert_equal(samples.loc[0], "sample_0")
            assert_equal(samples.loc[3], "sample_3")

            ploidy = bgen["genotype"][2].compute()["ploidy"]
            missing = bgen["genotype"][2].compute()["missing"]
            nvariants = len(variants)
            phased = [bgen["genotype"][i].compute()["phased"] for i in range(nvariants)]
            assert_allclose(ploidy, [1, 2, 2, 2])
            assert_allclose(missing, [0, 0, 0, 0])
            assert_allclose(phased, [0, 1, 1, 0, 1, 1, 1, 1, 0, 0])


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
        #sid_batch_size = 25
        #iid_count = 1
        #sid_count = 1*1000*1000
        #sid_batch_size = sid_count//10000
        iid_count = 2500
        sid_count = 100
        sid_batch_size = 10

        from pysnptools.distreader import DistGen
        from pysnptools.distreader import Bgen
        distgen = DistGen(seed=332,iid_count=iid_count,sid_count=sid_count,sid_batch_size=sid_batch_size)
        chrom_list = sorted(set(distgen.pos[:,0]))
        len(chrom_list)

        import logging
        logging.basicConfig(level=logging.INFO)
        #for chrom in chrom_list[::-1]:
        #    chromgen = distgen[:,distgen.pos[:,0]==chrom]
        chromgen = distgen

        print(chromgen.sid_count)
        name = '{0}x{1}'.format(iid_count,sid_count)
        gen_file = r'm:\deldir\{0}.gen'.format(name)
        sample_file2 = r'/mnt/m/deldir/{0}.sample'.format(name)
        gen_file2 = r'/mnt/m/deldir/{0}.gen'.format(name)
        print("about to read {0}x{1}".format(chromgen.iid_count,chromgen.sid_count))
        Bgen.genwrite(gen_file,chromgen,decimal_places=5,sid_batch_size=sid_batch_size) #better in batches?
        print("done")
        bgen_file = r'm:\deldir\{0}.bgen'.format(name)
        bgen_file2 = r'/mnt/m/deldir/{0}.bgen'.format(name)
        print ('/mnt/m/qctool/build/release/qctool_v2.0.7 -g {0} -s {1} -og {2} -bgen-bits 8 -bgen-compression zlib'.format(gen_file2,sample_file2,bgen_file2))

    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True)
    r.run(suites)

    import doctest
    result = doctest.testmod()
    assert result.failed == 0, "failed doc test: " + __file__
