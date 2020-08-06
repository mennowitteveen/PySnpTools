import numpy as np 


cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "./CPlinkBedFile.h":


	void _ImputeAndZeroMeanSNPsfloatFAAA "ImputeAndZeroMeanSNPsfloatFAAA"( 
		float *SNPs,
		size_t nIndividuals,
		size_t nSNPs,
		const bool betaNotUnitVariance,
		const float betaA,
		const float betaB,
		const bool apply_in_place,
		const bool use_stats,
		float *stats
		)
	void _ImputeAndZeroMeanSNPsdoubleFAAA "ImputeAndZeroMeanSNPsdoubleFAAA"( 
		double *SNPs,
		size_t nIndividuals,
		size_t nSNPs,
		const bool betaNotUnitVariance,
		const double betaA,
		const double betaB,
		const bool apply_in_place,
		const bool use_stats,
		double *stats
		)
	void _ImputeAndZeroMeanSNPsfloatCAAA "ImputeAndZeroMeanSNPsfloatCAAA"( 
		float *SNPs,
		size_t nIndividuals,
		size_t nSNPs,
		bool betaNotUnitVariance,
		float betaA,
		float betaB,
		const bool apply_in_place,
		const bool use_stats,
		float *stats
		)

	void _ImputeAndZeroMeanSNPsdoubleCAAA "ImputeAndZeroMeanSNPsdoubleCAAA"( 
		double *SNPs,
		size_t nIndividuals,
		size_t nSNPs,
		const bool betaNotUnitVariance,
		const double betaA,
		const double betaB,
		const bool apply_in_place,
		const bool use_stats,
		double *stats
		)


def standardizefloatFAAA(np.ndarray[np.float32_t, ndim=2] out, bool betaNotUnitVariance, float betaA, float betaB, bool apply_in_place, bool use_stats, np.ndarray[np.float32_t, ndim=2] stats):
	
	num_ind = out.shape[0]
	num_snps = out.shape[1]

	#http://wiki.cython.org/tutorials/NumpyPointerToC
	_ImputeAndZeroMeanSNPsfloatFAAA(<float*> out.data, num_ind, num_snps, betaNotUnitVariance, betaA, betaB, apply_in_place, use_stats, <float *> stats.data)

	return out, stats



def standardizedoubleFAAA(np.ndarray[np.float64_t, ndim=2] out, bool betaNotUnitVariance, double betaA, double betaB, bool apply_in_place, bool use_stats, np.ndarray[np.float64_t, ndim=2] stats):
	
	num_ind = out.shape[0]
	num_snps = out.shape[1]

	#http://wiki.cython.org/tutorials/NumpyPointerToC
	_ImputeAndZeroMeanSNPsdoubleFAAA(<double*> out.data, num_ind, num_snps, betaNotUnitVariance, betaA, betaB, apply_in_place, use_stats, <double *> stats.data)

	return out, stats



def standardizefloatCAAA(np.ndarray[np.float32_t, ndim=2] out, bool betaNotUnitVariance, float betaA, float betaB, bool apply_in_place, bool use_stats, np.ndarray[np.float32_t, ndim=2] stats):
	
	num_ind = out.shape[0]
	num_snps = out.shape[1]

	#http://wiki.cython.org/tutorials/NumpyPointerToC
	_ImputeAndZeroMeanSNPsfloatCAAA(<float*> out.data, num_ind, num_snps, betaNotUnitVariance, betaA, betaB, apply_in_place, use_stats, <float *> stats.data)

	return out, stats

def standardizedoubleCAAA(np.ndarray[np.float64_t, ndim=2] out, bool betaNotUnitVariance, double betaA, double betaB,  bool apply_in_place, bool use_stats, np.ndarray[np.float64_t, ndim=2] stats):
	
	num_ind = out.shape[0]
	num_snps = out.shape[1]

	#http://wiki.cython.org/tutorials/NumpyPointerToC
	_ImputeAndZeroMeanSNPsdoubleCAAA(<double*> out.data, num_ind, num_snps, betaNotUnitVariance, betaA, betaB, apply_in_place, use_stats, <double *> stats.data)

	return out, stats