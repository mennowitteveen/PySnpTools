import logging
import numpy as np
from pysnptools.pstreader import PstReader


#!!!cmk update documentation
class PstMemMap(PstReader):


    def __init__(self, filename):
        '''
        filename    : string of the name of the memory mapped file.
        '''
        #!!!there is also a NPZ file the way that bed has multiple files
        #!!!should/could they be one file using memmaps offset feature?

        super(PstMemMap, self).__init__()
        self._ran_once = False

        self._filename = filename

    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self._filename)

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

    def run_once(self):
        from pysnptools.snpreader import SnpReader

        if (self._ran_once):
            return
        self._ran_once = True

        npzfile = SnpReader._name_of_other_file(self._filename,"dat","npz")
        logging.debug("np.load('{0}')".format(npzfile))
        with np.load(npzfile) as data: #!! similar code in epistasis
            self._row = data['row']
            self._col = data['col']
            if np.array_equal(self._row, self._col): #If it's square, mark it so by making the col and row the same object
                self._col = self._row
            self._row_property = data['row_property']
            self._col_property = data['col_property']
            self._dtype = data['dtype'][0]
            self._order = data['order'][0]

        return self

    def copyinputs(self, copier):
        # doesn't need to self.run_once()
        copier.input(self._filename)
        npzfile = SnpReader._name_of_other_file(self._filename,"dat","npz")
        copier.input(npzfile)

    # Most _read's support only indexlists or None, but this one supports Slices, too.
    _read_accepts_slices = True
    def _read(self, row_index_or_none, col_index_or_none, order, dtype, force_python_only, view_ok):
        assert view_ok, "Expect view_ok to be True" #!!! good assert?
        self.run_once()
        mode = "r" if view_ok else "w+"
        logging.debug("val = np.memmap('{0}', dtype={1}, mode={2}, order={3}, shape=({4},{5}))".format(self._filename,self._dtype,mode,self._order,self.row_count,self.col_count))
        val = np.memmap(self._filename, dtype=self._dtype, mode=mode, order=self._order, shape=(self.row_count,self.col_count))
        val, _ = self._apply_sparray_or_slice_to_val(val, row_index_or_none, col_index_or_none, self._order, self._dtype, force_python_only) #!!! must confirm that this doesn't copy of view_ok
        return val

    @staticmethod
    def write(filename, pstdata):
        from pysnptools.snpreader import SnpMemMap

        npzfile = SnpReader._name_of_other_file(filename,"dat","npz")
        if pstdata.val.flags['F_CONTIGUOUS']:
            order = "F"
        elif pstdata.val.flags['C_CONTIGUOUS']:
            order = "C"
        else:
            raise Exception("Don't know order of PstData's value")

        np.savez(npzfile, row=pstdata.row, col=pstdata.col, row_property=pstdata.row_property, col_property=pstdata.col_property,dtype=np.array([pstdata.val.dtype]),order=np.array([order]))
        if isinstance(pstdata.val,np.memmap):
            pstdata.val.flush()
        else:
            val = np.memmap(filename, dtype=pstdata.val.dtype, mode="w+", order=order, shape=(pstdata.row_count,pstdata.col_count))
            val[:,:] = pstdata.val
            val.flush()
        logging.debug("Done writing " + filename)

        return SnpMemMap(filename) #!!! shouldn't all writers in pysnpsdata return their reader

