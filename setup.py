import platform
import os
import sys
import shutil
from setuptools import setup
import numpy

# Version number
version = '0.4.21'

def readme():
    with open('README.md') as f:
        return f.read()

install_requires = ['scipy>=1.1.0', 'numpy>=1.11.3', 'pandas>=0.19.0', 'psutil>=5.6.3', 'h5py>=2.10.0', 'dill>=0.2.9',
                   'backports.tempfile>=1.0', 'bgen-reader==4.0.5', 'bed-reader']

#python setup.py sdist bdist_wininst upload
setup(
    name='pysnptools',
    version=version,
    description='PySnpTools',
    long_description=readme(),
    long_description_content_type = 'text/markdown',
    keywords='gwas bioinformatics sets intervals ranges regions',
    url="https://fastlmm.github.io/",
    author='FaST-LMM Team',
    author_email='fastlmm-dev@python.org',
    license='Apache 2.0',
    classifiers = [
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python",
            ],
    packages=[  #basically everything with a __init__.py
        "pysnptools",
        "pysnptools/kernelreader",
        "pysnptools/kernelstandardizer",
        "pysnptools/pstreader",
        "pysnptools/snpreader",
        "pysnptools/distreader",
        "pysnptools/standardizer",
        "pysnptools/util",
        "pysnptools/util/filecache",
        "pysnptools/util/mapreduce1",
        "pysnptools/util/mapreduce1/runner",
    ],
    package_data={"pysnptools" : [
        "util/pysnptools.hashdown.json",
        "tests/mintest.py",
        ]
                 },

    install_requires = install_requires,
  )
