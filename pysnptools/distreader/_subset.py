from pysnptools.distreader import DistReader
from pysnptools.pstreader._subset import _PstSubset

class _DistSubset(_PstSubset,DistReader):
    def __init__(self, *args, **kwargs):
        super(_DistSubset, self).__init__(*args, **kwargs)
