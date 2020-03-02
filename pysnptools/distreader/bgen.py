import os
import logging
import numpy as np
from pysnptools.distreader import DistReader
import warnings
import unittest
from bgen_reader import read_bgen
from bgen_reader import example_files
from bgen_reader import create_metafile
from pysnptools.util import log_in_place
from os import remove
from shutil import move

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
    def __init__(self, filename, verbose=False, metadata=None, metadata2=None, double_iid_function=None, sid_function='id'):
        super(Bgen, self).__init__()
        self._ran_once = False
        self.read_bgen = None

        if double_iid_function is None:
            double_iid_function = lambda single_iid: ('',single_iid)

        if metadata2 is None: #!!!cmk in the case where it is not None, need to be sure it ends npz
            metadata2 = filename + ".metadata.npz"

        self.filename = filename
        self._verbose = verbose
        self._metadata = metadata
        self._metadata2 = metadata2
        self._double_iid_function = double_iid_function
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

        self._read_bgen = read_bgen(self.filename,verbose=self._verbose,metafile_filepath=self._metadata)

        if os.path.exists(self._metadata2):
            d = np.load(self._metadata2)
            self._row = d['row']
            self._col = d['col']
            self._col_property = d['col_property']
        else:
            logging.info("Reading and saving variant and sample metadata")


            self._row = self._col = np.array([self._double_iid_function(sample) for sample in self._read_bgen['samples']],dtype='str')

            if self._sid_function in {'id','rsid'}:
                self._col = np.array(self._read_bgen['variants'][self._sid_function],dtype='str') #!!!cmk23 test this
            else:
                id_list = self._read_bgen['variants']['id']
                rsid_list = self._read_bgen['variants']['rsid']
                self._col = np.array([self._sid_function(id,rsid) for id,rsid in zip(id_list,rsid_list)],dtype='str') #!!!cmk23 test this

            if len(self._col)>0: #spot check
                assert list(self._read_bgen['variants'].loc[0, "nalleles"])[0]==2, "Expect nalleles==2" #!!!cmk23 test that this is fast even with 1M sid_count

            self._col_property = np.zeros((len(self._col),3),dtype='float')
            self._col_property[:,0] = self._read_bgen['variants']['chrom']
            self._col_property[:,2] = self._read_bgen['variants']['pos']
            np.savez(self._metadata2,row=self._row,col=self._col,col_property=self._col_property)

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
        if self._ran_once:
            if hasattr(self,'_read_bgen') and self._read_bgen is not None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
                del self._read_bgen #!!!cmk is this needed? Meaningful?
                self._read_bgen = None
            self._ran_once = False


    #!!!cmk confirm that na na na 0,0,0 round trips
    #!!!cmk write a test for this
    @staticmethod
    def genwrite(filename, distreader, decimal_places=None, snpid_function=None, rsid_function=None, single_iid_function=None, sid_batch_size=100):
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

        snpid_function = snpid_function or (lambda sid:sid)
        rsid_function = rsid_function or (lambda sid:sid)
        single_iid_function = single_iid_function or (lambda f,i:i)

        if decimal_places is None:
            format_function = lambda num:'{0}'.format(num)
        else:
            format_function = lambda num:('{0:.'+str(decimal_places)+'f}').format(num)

        start = 0
        row_ascii = np.array(distreader.row,dtype='S') #!!! would be nice to avoid this copy when not needed.
        col_ascii = np.array(distreader.col,dtype='S') #!!! would be nice to avoid this copy when not needed.
        updater_freq = max(distreader.row_count * distreader.col_count // 100,1)
        with log_in_place("sid_index ", logging.INFO) as updater:
            with open(filename+'.temp','w',newline='\n') as genfp:
                while start < distreader.sid_count:
                    distdata = distreader[:,start:start+sid_batch_size].read(view_ok=True)
                    for sid_index in range(distdata.sid_count):
                        if sid_index % updater_freq == 0: #!!!cmk23 test this
                            updater('{0:,} of {1:,}'.format(start+sid_index,distreader.sid_count))#!!!cmk23 print commas in numbers test this
                        genfp.write('{0} {1} {2} {3} A G'.format(int(distdata.pos[sid_index,0]),snpid_function(distdata.sid[sid_index]),rsid_function(distdata.sid[sid_index]),int(distdata.pos[sid_index,2])))
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
                samplefp.write('{0}\n'.format(single_iid_function(f,i)))

        if os.path.exists(filename):
            remove(filename)
        move(filename+'.temp',filename)

class TestBgen(unittest.TestCase):    #!!!cmk23 be sure these are run

    def test1(self):
        logging.info("in TestBgen test1")
        bgen = Bgen('../examples/example.bgen')
        #print(bgen.iid,bgen.sid,bgen.pos)
        distdata = bgen.read()
        #print(distdata.val)
        bgen2 = bgen[:2,::3]
        #print(bgen2.iid,bgen2.sid,bgen2.pos)
        distdata2 = bgen2.read()
        #print(distdata2.val)

        #This and many of the tests based on bgen-reader-py\bgen_reader\test
    def test_bgen_samples_inside_bgen(self):
        with example_files("haplotypes.bgen") as filepath:
            data = Bgen(filepath)
            samples = [("","sample_0"), ("","sample_1"), ("","sample_2"), ("","sample_3")]
            assert (data.iid == samples).all()


    def test_bgen_samples_not_present(self):
        with example_files("complex.23bits.no.samples.bgen") as filepath:
            data = Bgen(filepath)
            samples = [("","sample_0"), ("","sample_1"), ("","sample_2"), ("","sample_3")]
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
    #def test_bgen_reader_phased_genotype(self): #!!!cmk think about support for phased
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



    def test_bgen_reader_variants_info(self):
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

            assert((bgen.iid[0] == ("", "sample_001")).all())
            assert((bgen.iid[7] == ("", "sample_008")).all())
            assert((bgen.iid[-1] == ("", "sample_500")).all())

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


    def test_bgen_reader_with_wrong_metadata_file(self):
        with example_files(["example.32bits.bgen", "wrong.metadata"]) as filepaths:
            bgen = Bgen(filepaths[0], metadata=filepaths[1])
            try:
                bgen.iid # expect error
                got_error = False
            except Exception as e:
                got_error = True
            assert got_error


    def cmktest_bgen_reader_with_nonexistent_metadata_file(self):
        with example_files("example.32bits.bgen") as filepath:
            folder = os.path.dirname(filepath)
            metafile_filepath = os.path.join(folder, "nonexistent.metadata")

            bgen = Bgen(filepath,metadata=metafile_filepath)
            bgen.iid
            assert os.path.exists(metafile_filepath)


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
