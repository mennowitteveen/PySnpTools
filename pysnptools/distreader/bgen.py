import os
import logging
import numpy as np
from pysnptools.distreader import DistReader
import warnings
import unittest
from bgen_reader import read_bgen
from bgen_reader import example_files
from bgen_reader import create_metafile

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
    def __init__(self, filename, verbose=False, metadata=None):
        super(Bgen, self).__init__()
        self._ran_once = False
        self.read_bgen = None
        self.filename = filename
        self._verbose = verbose
        self._metadata = metadata


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

        if self._metadata is None or self._metadata is False:
            metadata_file = None  #!!!cmk test this
        else:
            if self._metadata is True:
                metadata_file = self.filename + ".metadata" #!!!cmk test this
            else:
                metadata_file = self._metadata  #!!!cmk test this
            if not os.path.exists(metadata_file):
                create_metafile(self.filename, metadata_file, verbose=self._verbose)  #!!!cmk test this

        self._read_bgen = read_bgen(self.filename,verbose=self._verbose,metafile_filepath=metadata_file)

        if not hasattr(self,"_row"):
            #!!!cmk why no family id??? OK to to fill with blank??
            self._row = self._col = np.array([('',sample) for sample in self._read_bgen['samples']],dtype='str')

        if not hasattr(self,"_col") or not hasattr(self,"_col_property"):
            self._col = np.array(self._read_bgen['variants']['id'],dtype='str')
            self._col_property = np.zeros((len(self._col),3),dtype='float')
            self._col_property[:,0] = self._read_bgen['variants']['chrom'] #!!!cmk what if chrom not given? Is that possible? What about on write?
            self._col_property[:,1] = self._read_bgen['variants']['pos']
            #self._col_property[:,2] = bgen['variants']['chrom'] #!!!what is the 3rd value is 2nd right??? cmk
            #!!!!cmk might users want rsid instead of id???
            #!!!cmk should assert that nalleles==2 everywhere (is 1 OK?)
            print('cmk')

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
            probs = genotype[index_in].compute()['probs']
            val[:,index_out,:] = (probs[iid_index_out,:] if iid_index_or_none is not None else probs)
        
        return val


    def __del__(self):#!!!cmk
        if hasattr(self,'_read_bgen') and self._read_bgen is not None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
            del self._read_bgen #!!!cmk is this needed? Meaningful?
            self._read_bgen = None


    @staticmethod
    def write(filename, distdata, hdf5_dtype=None, sid_major=True): #!!!cmk
        """Writes a :class:`DistData` to DistHdf5 format and return a the :class:`.DistHdf5`.

        :param filename: the name of the file to create
        :type filename: string
        :param distdata: The in-memory data that should be written to disk.
        :type distdata: :class:`DistData`
        :param hdf5_dtype: None (use the .val's dtype) or a Hdf5 dtype, e.g. 'f8','f4',etc.
        :type hdf5_dtype: string
        :param sid_major: Tells if vals should be stored on disk in sid_major (default) or iid_major format.
        :type col_major: bool
        :rtype: :class:`.DistHdf5`

        >>> from pysnptools.distreader import DistHdf5, DistNpz
        >>> import pysnptools.util as pstutil
        >>> distdata = DistNpz('../examples/toydata.dist.npz')[:,:10].read()     # Read first 10 snps from DistNpz format
        >>> pstutil.create_directory_if_necessary("tempdir/toydata10.dist.hdf5")
        >>> DistHdf5.write("tempdir/toydata10.dist.hdf5",distdata)        # Write data in DistHdf5 format
        DistHdf5('tempdir/toydata10.dist.hdf5')
        """
        PstHdf5.write(filename,distdata,hdf5_dtype=hdf5_dtype,col_major=sid_major)
        return DistHdf5(filename)

class TestBgen(unittest.TestCase):    #!!!cmk be sure these are run

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


    def cmktest_bgen_reader_phased_genotype(self): #!!!cmk think about support for phased
        with example_files("haplotypes.bgen") as filepath:
            bgen = Bgen(filepath, verbose=False)
            assert(bgen.pos[0,0] == 1)
            assert(bgen.sid[0] == "SNP1")
            assert(bgen.pos[0,1]== 1)

            assert(bgen.pos[2,0] == 1)
            assert(bgen.sid[2] == "SNP3")
            assert(bgen.pos[2,1]== 3)

            assert((bgen.iid[0] ==("","sample_0")).all())
            assert((bgen.iid[2] ==("","sample_2")).all())

            assert((bgen.iid[-1] ==("","sample_3")).all())

            g = bgen[0,0].read()
            assert_allclose(g.val, [[[1.0, 0.0, 1.0, 0.0]]]) #cmk code doesn't know about phased
            g = bgen[-1,-1].read()
            assert_allclose(g.val, [[[1.0, 0.0, 0.0, 1.0]]])



    def test_bgen_reader_variants_info(self):
        with example_files("example.32bits.bgen") as filepath:
            bgen = Bgen(filepath)

            assert(bgen.pos[0,0]==1)
            assert(bgen.sid[0]=="SNPID_2")
            assert(bgen.pos[0,1] == 2000)

            assert(bgen.pos[7,0]==1)
            assert(bgen.sid[7]=="SNPID_9")
            assert(bgen.pos[7,1] == 9000)

            assert(bgen.pos[-1,0]==1)
            assert(bgen.sid[-1]=="SNPID_200")
            assert(bgen.pos[-1,1] == 100001)

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


    def test_bgen_reader_with_nonexistent_metadata_file(self):
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
                bgen = Bgen(filepath,metadata=True)
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

    if True: #!!!cmk
        pass
        #from bgen_reader import read_bgen
        #bgen = read_bgen('../examples/example.bgen')
        #bgen['variants']


    suites = getTestSuite()
    r = unittest.TextTestRunner(failfast=True)
    r.run(suites)

    import doctest
    result = doctest.testmod()
    assert result.failed == 0, "failed doc test: " + __file__
