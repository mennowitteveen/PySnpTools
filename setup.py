import platform
import os
import sys
import shutil
from setuptools import setup, Extension
from distutils.command.clean import clean as Clean
import numpy

# Version number
version = '0.4.1'

def readme():
    with open('README.md') as f:
        return f.read()

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

#use_cython=False

class CleanCommand(Clean):
    description = "Remove build directories, and compiled files (including .pyc)"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if (   filename.endswith('.so')
                    or filename.endswith('.pyd')
                    #or filename.find("wrap_plink_parser.cpp") != -1 # remove automatically generated source file
                    #or filename.find("wrap_matrix_subset.cpp") != -1 # remove automatically generated source file
                    or filename.endswith('.pyc')
                                ):
                    tmp_fn = os.path.join(dirpath, filename)
                    print("removing", tmp_fn)
                    os.unlink(tmp_fn)

# set up macro
if platform.system() == "Darwin":
    macros = [("__APPLE__", "1")]
elif "win" in platform.system().lower():
    macros = [("_WIN32", "1")]
else:
    macros = [("_UNIX", "1")]


#see http://stackoverflow.com/questions/4505747/how-should-i-structure-a-python-package-that-contains-cython-code
if use_cython:
    ext_modules = [Extension(name="pysnptools.snpreader.wrap_plink_parser",
                             language="c++",
                             sources=["pysnptools/snpreader/wrap_plink_parser.pyx", "pysnptools/snpreader/CPlinkBedFile.cpp"],
                             include_dirs = [numpy.get_include()],
                             define_macros=macros),
                   Extension(name="pysnptools.snpreader.wrap_matrix_subset",
                            language="c++",
                            sources=["pysnptools/snpreader/wrap_matrix_subset.pyx", "pysnptools/snpreader/MatrixSubset.cpp"],
                            include_dirs = [numpy.get_include()],
                            define_macros=macros)]
    cmdclass = {'build_ext': build_ext, 'clean': CleanCommand}
else:
    ext_modules = [Extension(name="pysnptools.snpreader.wrap_plink_parser",
                             language="c++",
                             sources=["pysnptools/snpreader/wrap_plink_parser.cpp", "pysnptools/snpreader/CPlinkBedFile.cpp"],
                             include_dirs = [numpy.get_include()],
                             define_macros=macros),
                   Extension(name="pysnptools.snpreader.wrap_matrix_subset",
                            language="c++",
                            sources=["pysnptools/snpreader/wrap_matrix_subset.cpp", "pysnptools/snpreader/MatrixSubset.cpp"],
                            include_dirs = [numpy.get_include()],
                            define_macros=macros)]
    cmdclass = {}



class CleanCommand(Clean):
    description = "Remove build directories, and compiled files (including .pyc)"

    def run(self):
        Clean.run(self)
        if os.path.exists('build'):
            shutil.rmtree('build')
        for dirpath, dirnames, filenames in os.walk('.'):
            for filename in filenames:
                if (   filename.endswith('.so')
                    or filename.endswith('.pyd')
                    or filename.find("wrap_plink_parser.cpp") != -1 # remove automatically generated source file
                    or filename.find("wrap_matrix_subset.cpp") != -1 # remove automatically generated source file
                    or filename.endswith('.pyc')
                                ):
                    tmp_fn = os.path.join(dirpath, filename)
                    print("removing", tmp_fn)
                    os.unlink(tmp_fn)

#python setup.py sdist bdist_wininst upload
setup(
    name='pysnptools',
    version=version,
    description='PySnpTools',
    long_description=readme(),
    keywords='gwas bioinformatics sets intervals ranges regions',
    url="https://fastlmm.github.io/",
    author='FaST-LMM Team',
    author_email='fastlmm-dev@python.org',
    license='Apache 2.0',
    packages=[  #basically everything with a __init__.py
        "pysnptools",
        "pysnptools/kernelreader",
        "pysnptools/kernelstandardizer",
        "pysnptools/pstreader",
        "pysnptools/snpreader",
        "pysnptools/standardizer",
        "pysnptools/util",
        "pysnptools/util/filecache",
        "pysnptools/util/mapreduce1",
        "pysnptools/util/mapreduce1/runner",
    ],
    package_data={"pysnptools" : [
        "test/datasets/all_chr.maf0.001.N300.bed",
        "test/datasets/all_chr.maf0.001.N300.bim",
        "test/datasets/all_chr.maf0.001.N300.fam",
        "tests/datasets/all_chr.maf0.001.covariates.N300.txt",
        "test/datasets/phenSynthFrom22.23.N300.randcidorder.txt",
        "tests/datasets/snpgen.bed",
        "tests/datasets/snpgen.bim",
        "tests/datasets/snpgen.fam",
        ]
                 },
    install_requires = ['scipy>=1.1.0', 'numpy>=1.11.3', 'pandas>=0.19.0'],

    # extensions
    cmdclass = cmdclass,
    ext_modules = ext_modules
  )

