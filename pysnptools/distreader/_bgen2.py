
import os
import numpy as np
from contextlib import contextmanager
from tempfile import mkdtemp
import shutil

from bgen_reader._ffi import ffi, lib
from bgen_reader._string import bgen_str_to_str
from bgen_reader._samples import get_samples
from bgen_reader._metadata import create_metafile
from bgen_reader._bgen import bgen_file, bgen_metafile
from bgen_reader._partition import _read_allele_ids #!!!cmk could just copy this one over


@contextmanager
def bgen_reader2(filename, sample=None, verbose=False):
    bgen2 = _Bgen2(filename,sample=sample,verbose=verbose)
    yield bgen2
    del bgen2


class _Bgen2(object):
    def __init__(self, filename, sample=None, verbose=False):
        self.filename = filename #_filename?
        self._sample = sample
        self._verbose = verbose

        assert os.path.exists(self.filename), "Expect file to exist ('{0}')".format(self.filename)
        self.samples = np.array(get_samples(self.filename,self._sample,verbose),dtype='str') #!!!cmk leave as pd.Series? go directly to numpy?
        self._bgen_context_manager = bgen_file(self.filename)
        self._bgen = self._bgen_context_manager.__enter__()

        metadata2 = self.filename + ".metadata2.npz"
        if os.path.exists(metadata2):
            d = np.load(metadata2) #!!!cmk could do memory mapping instead
            self.id = d['id']
            self.rsid = d['rsid']
            self.vaddr = d['vaddr']
            self.chrom = d['chrom']
            self.position = d['position']
            self.nalleles = d['nalleles']
            self.allele_ids = d['allele_ids']
            self.ncombs = d['ncombs']
            self.phased = d['phased']
        else:
            tempdir = None
            try:
                tempdir = mkdtemp(prefix='pysnptools')
                metafile_filepath = tempdir+'/bgen.metadata'
                create_metafile(self.filename,metafile_filepath,verbose=verbose)
                self._map_metadata(metafile_filepath)
                np.savez(metadata2,id=self.id,rsid=self.rsid,vaddr=self.vaddr,chrom=self.chrom,position=self.position,
                         nalleles=self.nalleles,allele_ids=self.allele_ids,ncombs=self.ncombs,phased=self.phased)
            finally:
                if tempdir is not None:
                    shutil.rmtree(tempdir)

        self.max_ncombs = max(self.ncombs)

    #!!!cmk add an nvariants property and nsamples
    #!!!cmk should have dtype (because float32 is often enough and is 1/2 the size) and order
    def read(self, variants=None, max_ncombs=None): #!!!cmk also allow samples to be selected?
        #!!!cmk allow single ints, lists of ints, lists of bools, None, and slices
        #!!!cmk could allow strings (variant names) and lists of strings

        max_ncombs = max_ncombs or self.max_ncombs

        if type(variants) is np.int: #!!!make sure this works with all int types
            variants = [variants]
        if variants is None:
            vaddr = self.vaddr
            ncombs = self.ncombs
        else:
            vaddr = self.vaddr[variants]
            ncombs = self.ncombs[variants]

        #allocating p only once make reading 10x5M data 30% faster
        val = np.full((len(self.samples), len(vaddr), max_ncombs), np.nan, order='F', dtype='float64') #!!!cmk test on selecting zero variants
        p = None

        #LATER multithread?
        #!!!cmk if verbose is true, give some status
        for out_index,vaddr0 in enumerate(vaddr):
            if p is None or ncombs[out_index] != p.shape[-1]:
                p = np.full((len(self.samples), ncombs[out_index]), np.nan, order='C', dtype='float64')
            vg = lib.bgen_open_genotype(self._bgen, vaddr0)
            lib.bgen_read_genotype(self._bgen, vg, ffi.cast("double *", p.ctypes.data))
            #ploidy = asarray([lib.bgen_ploidy(vg, i) for i in range(nsamples)], int) #!!!cmk what is this? It will likely be a different call
            #missing = asarray([lib.bgen_missing(vg, i) for i in range(nsamples)], bool) #!!!cmk why is this need instead of just seeing nan,nan,nan
            lib.bgen_close_genotype(vg)
            val[:,out_index,:ncombs[out_index]] = p
        return val

    def _map_metadata(self,metafile_filepath): 
        with bgen_metafile(metafile_filepath) as mf:
            nparts = lib.bgen_metafile_npartitions(mf)
            id_list, rsid_list,chrom_list,position_list,vaddr_list,nalleles_list,allele_ids_list,ncombs_list,phased_list = [],[],[],[],[],[],[],[],[]

            #!!!If verbose, should tell how it is going
            for ipart in range(nparts): #LATER multithread?
                nvariants_ptr = ffi.new("int *")
                metadata = lib.bgen_read_partition(mf, ipart, nvariants_ptr)
                nvariants_in_part = nvariants_ptr[0]
                for i in range(nvariants_in_part):
                    if self._verbose and len(id_list)%1000==0:
                        print('{0}'.format(len(id_list)))
                    nalleles_list.append(metadata[i].nalleles)
                    id_list.append(bgen_str_to_str(metadata[i].id))
                    rsid_list.append(bgen_str_to_str(metadata[i].rsid))
                    chrom_list.append(bgen_str_to_str(metadata[i].chrom))
                    position_list.append(metadata[i].position)
                    vaddr0 = metadata[i].vaddr
                    vaddr_list.append(vaddr0)
                    allele_ids_list.append(_read_allele_ids(metadata[i])) #!!!cmk Should check that didn't already have ,???
                    vg = lib.bgen_open_genotype(self._bgen, vaddr0)
                    ncombs_list.append(lib.bgen_ncombs(vg))
                    phased_list.append(lib.bgen_phased(vg)) #!!!cmk is it really the case that some variants can be phased and others not?
                    lib.bgen_close_genotype(vg)

        self.id = np.array(id_list,dtype='str')
        self.rsid = np.array(rsid_list,dtype='str')
        self.vaddr = np.array(vaddr_list,dtype=np.uint64)#!!!cmk99 Wait for fix
        self.chrom = np.array(chrom_list,dtype='str')
        self.position = np.array(position_list,dtype=np.int) #!!!cmk check int not unit, etc
        self.nalleles = np.array(nalleles_list,dtype=np.int)
        self.allele_ids = np.array(nalleles_list,dtype='str')
        self.ncombs = np.array(ncombs_list,dtype='int')
        self.phased = np.array(phased_list,dtype='bool')

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.filename)

    def __del__(self):
        if hasattr(self,'_bgen_context_manager') and self._bgen_context_manager is not None:  # we need to test this because Python doesn't guarantee that __init__ was fully run
            self._bgen_context_manager.__exit__(None,None,None)

if __name__ == "__main__":
    if False:
        #filename = r'm:\deldir\1000x500000.bgen'
        filename = r'D:\OneDrive\Shares\bgenreaderpy\1x1000000.bgen'
        with bgen_reader2(filename) as bgen2:
            print(bgen2.id[:5]) #other property arrays include risd,chrom,position,nallels, and allele_ids
            geno = bgen2.read(199999) # read the 200,000th variate's data
            #geno = bgen2.read() # read all, uses the ncombs from the first variant
            geno = bgen2.read(slice(5)) # read first 5, uses the ncombs from the first variant
            geno = bgen2.read(bgen2.chrom=='5',max_ncombs=4) # read chrom1, set max_combs explicitly
    if True:
        filename = r'm:\deldir\2500x500000.bgen'
        with bgen_reader2(filename,verbose=True) as bgen2:
            print(bgen2.reader(0)[0,0,:])
            print(bgen2.reader(-1)[0,0,:])
    print('!!!done')
