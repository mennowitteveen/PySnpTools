
import os
import fnmatch
from pysnptools.util.filecache import Hashdown
import logging

#!!!cmk later replace with hashdown.json

if False:
    hashdown_local = Hashdown.scan_local(r'D:\OneDrive\programs\pysnptools',
                                         url='https://github.com/fastlmm/PySnpTools/raw/cf248cbf762516540470d693532590a77c76fba2').save_hashdown(
                                             'deldir/pysnptools.hashdown.json',verbose=True)

pysnptools_hashdown = Hashdown.load_hashdown(os.path.join(os.path.dirname(os.path.realpath(__file__)),"pysnptools.hashdown.json"))

if False:
    update = Hashdown(url=pysnptools_hashdown.url,allow_unhashed_files=True)
    for file in pysnptools_hashdown.file_to_hash:
        print(file)
        update.file_exists(file)
    update.save_hashdown('deldir/updated.hashdown.json')

#!!!cmk should do something to check that hashcodes are up-to-date
def example_file(pattern,endswith=None):
    #!!!cmk add credit to Danilo for idea
    #!!!cmk add env var to allow to move to place that already has the files

    return_file = None
    for filename in fnmatch.filter(pysnptools_hashdown.file_to_hash, pattern):
        with pysnptools_hashdown.open_read(filename) as local_file:
            if return_file is None and (endswith is None or fnmatch.fnmatch(filename,endswith)):
                return_file = local_file
    assert return_file is not None, "Pattern not found '{0}'{1}".format(pattern,'' if (endswith is None) else "'{0}'".format(endswith))
    return return_file

