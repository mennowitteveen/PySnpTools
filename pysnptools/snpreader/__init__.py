"""Tools for reading and manipulating SNP data.
"""

from pysnptools.snpreader.snpreader import SnpReader
from pysnptools.snpreader.snpdata import SnpData
from pysnptools.snpreader.bed import Bed
from pysnptools.snpreader.ped import Ped
from pysnptools.snpreader.dat import Dat
from pysnptools.snpreader.snphdf5 import SnpHdf5
from pysnptools.snpreader.snphdf5 import Hdf5
from pysnptools.snpreader.snpnpz import SnpNpz
from pysnptools.snpreader.dense import Dense
from pysnptools.snpreader.pheno import Pheno
from pysnptools.snpreader.pairs import Pairs
from pysnptools.snpreader.snpmemmap import SnpMemMap
from pysnptools.snpreader._mergesids import _MergeSIDs
from pysnptools.snpreader._mergeiids import _MergeIIDs
from pysnptools.snpreader.distributed_bed import DistributedBed #Must be after _MergeSIDs and _MergeIIDs
