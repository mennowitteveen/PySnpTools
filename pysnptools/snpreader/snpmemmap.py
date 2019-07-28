import logging
import os
from pysnptools.pstreader import PstMemMap
from pysnptools.snpreader import SnpReader, SnpData
import numpy as np

#!!!cmk update documentation
class SnpMemMap(PstMemMap,SnpReader):

    def __init__(self, *args, **kwargs):
        super(SnpMemMap, self).__init__(*args, **kwargs)

    @staticmethod
    def write(filename, snpdata):
        return PstMemMap.write(filename,snpdata)

    @staticmethod #!!!PstMemMap should have something like this, too
    def snp_data(iid,sid,filename,pos=None,order="F",dtype=np.float64):
        shape = (len(iid),len(sid))
        logging.info("About to start allocating memmap '{0}'".format(filename))
        fp = np.memmap(filename, dtype=dtype, mode="w+", order=order, shape=shape)
        logging.info("Finished allocating memmap '{0}'. Size is {1}".format(filename,os.path.getsize(filename)))
        result = SnpData(iid=iid,sid=sid,val=fp,pos=pos,name="np.memmap('{0}')".format(filename))
        return result

