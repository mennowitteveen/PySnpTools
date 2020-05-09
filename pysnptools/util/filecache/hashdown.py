import os
import shutil
import hashlib
try:
    from urllib.request import urlopen, HTTPError # Python 3
except ImportError:
    from urllib2 import urlopen, HTTPError # Python 2
import logging
from contextlib import contextmanager
import pysnptools.util as pstutil
from pysnptools.util.filecache import FileCache
import tempfile
import json
from pysnptools.util import log_in_place

class Hashdown(FileCache):
    '''
    A :class:`.FileCache` for working with files locally.#!!!cmk update

    See :class:`.FileCache` for general examples of using FileCache.

    This is the simplest :class:`.FileCache` in that it stores everything on a local disk rather than storing things remotely.

    **Constructor:**
        :Parameters: * **directory** (*string*) -- The directory under which files should be written and read.

        :Example:

        >>> #cmkfrom pysnptools.util.filecache import Hashdown
        >>> #file_cache = Hashdown('localcache1')
        >>> #file_cache.rmtree()
        >>> #file_cache.save('sub1/file1.txt','Hello')
        >>> #file_cache.file_exists('sub1/file1.txt')        True

    '''
    def __init__(self,url,file_to_hash={},directory=None,allow_unhashed_files=False,trust_local_files=False,relative_dir=None):
        super(Hashdown, self).__init__()
        self.url = url
        self.file_to_hash = file_to_hash
        self.allow_unhashed_files = allow_unhashed_files
        self.trust_local_files = trust_local_files #!!!cmk need to test this
        base_url = url if relative_dir is None else url[:-len(relative_dir)-1]
        url_hash = hashlib.md5(base_url.encode('utf-8')).hexdigest()
        self.directory =  tempfile.gettempdir() + '/hashdown/{0}'.format(url_hash)
        self.relative_dir = relative_dir
        if os.path.exists(self.directory): assert not os.path.isfile(self.directory), "A directory cannot exist where a file already exists."


    def __repr__(self): 
        return "{0}('{1}')".format(self.__class__.__name__,self.url)

    @property
    def name(self):
        '''
        A path-like name for this `LocalCache`.#!!!update

        :rtype: string
        '''
        return self.directory

    @staticmethod
    def _get_large_file(url, file, trust_local_files, length=16*1024):
        '''https://stackoverflow.com/questions/1517616/stream-large-binary-files-with-urllib2-to-file
        '''
        logging.info("Downloading'{0}'".format(url))
        if trust_local_files and os.path.exists(file):
            logging.info("Trusting local file'{0}'".format(file))
            return True

        try:
            req = urlopen(url)
        except HTTPError as e:
            if e.code == 404:
                return False
            raise

        pstutil.create_directory_if_necessary(file)
        with open(file, 'wb') as fp:
            shutil.copyfileobj(req, fp, length)
        return True

    @staticmethod
    def _get_hash(filename):
        '''https://stackoverflow.com/questions/16874598/how-do-i-calculate-the-md5-checksum-of-a-file-in-python
        '''
        logging.info("Find hash of '{0}'".format(filename))
        with open(filename, "rb") as f:
            file_hash = hashlib.md5()
            while True:
                chunk = f.read(8192)
                if chunk == b'':
                    break
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def _simple_file_exists(self,simple_file_name):
        rel_part = "" if self.relative_dir is None else self.relative_dir + "/"
        full_file = self.directory + "/" + rel_part + simple_file_name
        relative_file = simple_file_name if self.relative_dir is None else self.relative_dir + '/' + simple_file_name
        full_url = self.url + "/" + simple_file_name
        hash = self.file_to_hash.get(relative_file)

        if hash is not None:
            return True

        if not self.allow_unhashed_files:
            return False

        if self._get_large_file(full_url,full_file,self.trust_local_files): #!!!cmk this should overwrite any local file. #!!!cmk when did directory get created?
            hash = self._get_hash(full_file)
            self.file_to_hash[relative_file]=hash
            return True
        else:
            return False


    @contextmanager
    def _simple_open_read(self,simple_file_name,updater=None):
        logging.info("open_read('{0}')".format(simple_file_name))

        relative_file = simple_file_name if self.relative_dir is None else self.relative_dir + '/' + simple_file_name
        full_file = self.directory + "/" + relative_file
        full_url = self.url + "/" + simple_file_name
        hash = self.file_to_hash.get(relative_file)
        assert self._simple_file_exists(simple_file_name), "File doesn't exist ('{0}')".format(relative_file)
        if os.path.exists(full_file):
            local_hash = self._get_hash(full_file)
        else:
            local_hash = None

        if local_hash is None or local_hash != hash:
            assert self._get_large_file(full_url,full_file,trust_local_files=False), "URL return 'no item' ('{0}')".format(full_url)
            local_hash = self._get_hash(full_file)
            if hash is None:
                assert self.allow_unhashed_files, "real assert"
                self.file_to_hash[relative_file]=local_hash
            else:
                assert hash==local_hash, 'URL file has unexpected hash ("{0}")'.format(full_url)

        yield full_file

        logging.info("close('{0}')".format(simple_file_name))

    @contextmanager
    def _simple_open_write(self,simple_file_name,size=0,updater=None):
        raise ValueError('Hashdown is read only. writing is not allowed.')

    def _simple_rmtree(self,updater=None):
        raise ValueError('Hashdown is read only. "rmtree" is not allowed.')

    def _simple_remove(self,simple_file_name, updater=None):
        raise ValueError('Hashdown is read only. "remove" is not allowed.')

    def _simple_getmtime(self,simple_file_name):
        assert self._simple_file_exists(simple_file_name), "file doesn't exist ('{0}')".format(simple_file_name)
        return 0

    def _simple_join(self,path):
        directory = self.directory + "/" + path
        relative_dir = path if self.relative_dir is None else self.relative_dir + '/' + path
        if not self.allow_unhashed_files:
            assert not self.file_exists(relative_dir), "Can't treat an existing file as a directory"
        return Hashdown(url=self.url+'/'+path,directory=directory,file_to_hash=self.file_to_hash,allow_unhashed_files=self.allow_unhashed_files,
                        relative_dir=relative_dir)


    def _simple_walk(self):
        for rel_file in self.file_to_hash:
            if self.relative_dir is None or rel_file.startswith(self.relative_dir+"/"):
                file = rel_file if self.relative_dir is None else rel_file[len(self.relative_dir)+1:]
                if self.file_exists(file):
                    yield file

    def save_hashdown(self, filename):
        pstutil.create_directory_if_necessary(filename)
        dict0 = dict(self.__dict__)
        del dict0['directory']
        with open(filename, 'w') as json_file:
            json.dump(dict0,json_file)


    @staticmethod
    def load_hashdown(filename,directory=None):
        with open(filename) as json_file:
            dict0 = json.load(json_file)
        hashdown = Hashdown(url=dict0['url'],file_to_hash=dict0['file_to_hash'],directory=directory,allow_unhashed_files=dict0['allow_unhashed_files'],relative_dir=dict0['relative_dir'])
        return hashdown

    @staticmethod
    def scan_local(local_directory, url=None, logging_level=logging.WARNING):
        from pysnptools.util.filecache import LocalCache
        file_to_hash = {}
        localcache = LocalCache(local_directory)
        with log_in_place("scanning", logging_level) as updater:
            for file in localcache.walk():
                updater(file)
                with localcache.open_read(file) as full_file:
                    hash = Hashdown._get_hash(full_file)
                    file_to_hash[file] = hash
        return Hashdown(url,file_to_hash=file_to_hash)

if __name__ == "__main__":
    if True:
        hashdown_local = Hashdown.scan_local(r'D:\OneDrive\programs\fastlmm',url='https://github.com/fastlmm/FaST-LMM/raw/ff183d3aa09c78cf5fdf1961e9241e8a9b9dd172')
        hashdown_local.save_hashdown('deldir/fastlmm.hashdown.json')



    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()
