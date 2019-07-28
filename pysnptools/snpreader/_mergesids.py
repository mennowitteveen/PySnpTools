from pysnptools.snpreader import SnpReader
from pysnptools.pstreader import _MergeCols

class _MergeSIDs(_MergeCols,SnpReader): #!!! move to PySnptools
    def __init__(self,  *args, **kwargs):
        super(_MergeSIDs, self).__init__(*args, **kwargs)

