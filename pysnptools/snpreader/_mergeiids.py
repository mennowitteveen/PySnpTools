from pysnptools.snpreader import SnpReader
from pysnptools.pstreader import _MergeRows


class _MergeIIDs(_MergeRows,SnpReader): #!!! move to PySnptools
    def __init__(self, *args, **kwargs):
        super(_MergeIIDs, self).__init__(*args, **kwargs)
