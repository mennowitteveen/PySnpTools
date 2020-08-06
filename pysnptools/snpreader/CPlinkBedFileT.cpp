/*
*******************************************************************
*
*    Copyright (c) Microsoft. All rights reserved.
*
*    THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
*    ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
*    IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
*    PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
*
******************************************************************
*/

/*
* CPlinkBedFile - {PLINK BED File Access Class}
*
*         File Name:   CPlinkBedFile.cpp
*           Version:   2.00
*            Author:   
*     Creation Date:   18 Nov 2010
*     Revision Date:   14 Aug 2013
*
*    Module Purpose:   This file implements the CPlinkBedFile 
*  
*                      A .BED file contains compressed binary genotype 
*                         values for individuals by SNPs.  
*
*    Change History:   Version 2.00: Reworked to be wrapped in python version by Chris Widmer  (chris@shogun-toolbox.org)
*
* Test Files: 
*/

/*
* Include Files
*/
#include "CPlinkBedFileT.h"
#include <iostream>
#include <stdio.h>
#include <math.h> 
#include <stdlib.h>

#ifdef _WIN32
#define isinf(x) (!_finite(x))
#else
#define isinf(x) (!isfinite(x))
#endif

const REAL SUFFIX(_PI) = 2.0*acos(0.0);
const REAL SUFFIX(_halflog2pi)=(REAL)0.5*log((REAL)2.0*SUFFIX(_PI));
const REAL SUFFIX(coeffsForLogGamma)[] = { 12.0, -360.0, 1260.0, -1680.0, 1188.0 };

const REAL SUFFIX(eps_rank)=(REAL)3E-8;


// Gamma and LogGamma

// Use GalenA's implementation of LogGamma - it's faster!
/// <summary>Returns the log of the gamma function</summary>
/// <param name="x">Argument of function</param>
/// <returns>Log Gamma(x)</returns>
/// <remarks>Accurate to eight digits for all x.</remarks>
REAL SUFFIX(logGamma)(REAL x)
{
	if (x <= (REAL)0.0){
		printf("LogGamma arg=%f must be > 0.",x);
		throw(1);
	}

	REAL res = (REAL)0.0;
	if (x < (REAL)6.0)
	{
		int toAdd = (int)floor(7 - x);
		REAL v2 = (REAL)1.0;
		for (int i = 0; i < toAdd; i++)
		{
			v2 *= (x + i);
		}
		res = -log(v2);
		x += toAdd;
	}
	x -= (REAL)1.0;

	res += SUFFIX(_halflog2pi) + (x + (REAL)0.5) * log(x) - x;

	// correction terms
	REAL xSquared = x * x;
	REAL pow = x;
	for (int i=0; i<5; ++i)   //the length of the coefficient array is 5.
	{
		REAL newRes = res + (REAL)1.0 / (SUFFIX(coeffsForLogGamma)[i] * pow);
		if (newRes == res)
		{
			return res;
		}
		res = newRes;
		pow *= xSquared;
	}

	return res;
}

// Beta and LogBeta
/// <summary>Computes the log beta function</summary>
double SUFFIX(LogBeta)(REAL x, REAL y)
{
	if (x <= 0.0 || y <= 0.0){
		printf("LogBeta args must be > 0.");
		throw(1);
	}
	return SUFFIX(logGamma)(x) + SUFFIX(logGamma)(y) - SUFFIX(logGamma)(x + y);
}

/// <summary>Probability distribution function</summary>
/// <param name="x">Value at which to compute the pdf</param>
/// <param name="a">Shape parameter (alpha)</param>
/// <param name="b">Shape parameter (beta)</param>
REAL SUFFIX(BetaPdf)(REAL x, REAL a, REAL b){
   if (a <= 0 || b <= 0){
      printf("Beta.Pdf parameters, a and b, must be > 0");
      throw(1);
   }

   if (x > 1) return 0;
   if (x < 0) return 0;

   REAL lnb = SUFFIX(LogBeta)(a, b);
   return exp((a - 1) * log(x) + (b - 1) * log(1 - x) - lnb);
}



/*
* Parameters: 
* SNPs [nIndividuals by nSNPs]:
*                       Matrix stored in column-major order. 
*                       This will hold the result.
*                       NaNs will be set to 0.0 in the result.
*/
void SUFFIX(ImputeAndZeroMeanSNPs)( 
	REAL *SNPs, 
	const size_t nIndividuals, 
	const size_t nSNPs, 
	const bool betaNotUnitVariance,
	const REAL betaA,
	const REAL betaB,
	const bool apply_in_place,
	const bool use_stats,
	REAL *stats
	)
{
	bool seenSNC = false; //Keep track of this so that only one warning message is reported
#ifdef ORDERF

	for ( size_t iSnp = 0; iSnp < nSNPs; ++iSnp )
	{
		REAL mean_s;
		REAL std;
		size_t end = nIndividuals;
		size_t delta = 1;
		bool isSNC;

		if (use_stats)
		{
			mean_s = stats[iSnp];
			std = stats[iSnp + nSNPs];
			isSNC = isinf(std);
		}
		else
		{
			isSNC = false;
			REAL n_observed = 0.0;
			REAL sum_s = 0.0;      //the sum of a SNP over all observed individuals
			REAL sum2_s = 0.0;      //the sum of the squares of the SNP over all observed individuals

			for (size_t ind = 0; ind < end; ind += delta)
			{
				if (SNPs[ind] == SNPs[ind])
				{
					//check for not NaN
					sum_s += SNPs[ind];
					sum2_s += SNPs[ind] * SNPs[ind];
					++n_observed;
				}
			}

			if (n_observed < 1.0)
			{
				printf("No individual observed for the SNP.\n");
				//LATER make it work (in some form) for n of 0
			}

			mean_s = sum_s / n_observed;    //compute the mean over observed individuals for the current SNP
			REAL mean2_s = sum2_s / n_observed;    //compute the mean of the squared SNP

			if ((mean_s != mean_s) || (betaNotUnitVariance && ((mean_s > (REAL)2.0) || (mean_s < (REAL)0.0))))
			{
				if (!seenSNC)
				{
					seenSNC = true;
					fprintf(stderr, "Illegal SNP mean: %.2f for SNPs[:][%zu]\n", mean_s, iSnp);
				}
			}


			REAL variance = mean2_s - mean_s * mean_s;        //By the Cauchy Shwartz inequality this should always be positive
			std = sqrt(variance);

			if ((std != std) || (std <= (REAL)0.0))
			{
				// a std == 0.0 means all SNPs have the same value (no variation or Single Nucleotide Constant (SNC))
				//   however ALL SNCs should have been removed in previous filtering steps
				//   This test now prevents a divide by zero error below
				isSNC = true;
				if (!seenSNC)
				{
					seenSNC = true;
					//#Don't need this warning because SNCs are still meaning full in QQ plots because they should be thought of as SNPs without enough data.
					//fprintf(stderr, "std=.%2f has illegal value for SNPs[:][%zu]\n", std, iSnp);
				}
				std = std::numeric_limits<REAL>::infinity();

			}

			stats[iSnp] = mean_s;
			stats[iSnp + nSNPs] = std;
		}


		if (apply_in_place)
		{
			for (size_t ind = 0; ind < end; ind += delta)
			{
				//check for NaN
				if ((SNPs[ind] != SNPs[ind]) || isSNC)
				{
					SNPs[ind] = 0.0;
				}
				else
				{
					SNPs[ind] -= mean_s;     //subtract the mean from the data
					if (betaNotUnitVariance) //compute snp-freq as in the Visscher Height paper (Nat Gen, Yang et al 2010).
					{
						REAL freq = mean_s / 2.0;
						if (freq > .5)
						{
							freq = 1.0 - freq;
						}
						REAL rT = SUFFIX(BetaPdf)(freq, betaA, betaB);
						//fprintf(stderr, "BetaPdf(%f,%f,%f)=%f\n",  freq, betaA, betaB, rT);
						SNPs[ind] *= rT;
					}
					else
					{
						SNPs[ind] /= std;        //unit variance as well
					}
				}
			}
		}

		SNPs += nIndividuals;
	}

#else //Order C

	std::vector<REAL> mean_s(nSNPs);  //compute the mean over observed individuals for the current SNP
	std::vector<REAL> std(nSNPs); //the standard deviation
	std::vector<bool> isSNC(nSNPs); // Is this a SNC (C++ inits to false)
	if (use_stats)
	{
		for (size_t iSnp = 0; iSnp < nSNPs; ++iSnp)
		{
			mean_s[iSnp] = stats[iSnp*2];
			std[iSnp] = stats[iSnp * 2+1];
			isSNC[iSnp] = isinf(std[iSnp]);
		}
	}
	else
	{
		// Make one pass through the data (by individual, because that is how it is laid out), collecting statistics
		std::vector<REAL> n_observed(nSNPs); //                                                C++ inits to 0's
		std::vector<REAL> sum_s(nSNPs);      //the sum of a SNP over all observed individuals. C++ inits to 0's
		std::vector<REAL> sum2_s(nSNPs);     //the sum of the squares of the SNP over all observed individuals.     C++ inits to 0's

		for( size_t ind = 0; ind < nIndividuals; ++ind)
		{
			size_t rowStart = ind * nSNPs;
			for ( size_t iSnp = 0; iSnp < nSNPs; ++iSnp )
			{
				REAL value = SNPs[rowStart+iSnp];
				if ( value == value )
				{
					sum_s[iSnp] += value;
					sum2_s[iSnp] += value * value;
					++n_observed[iSnp];
				}
			}
		}


		std::vector<REAL> mean2_s(nSNPs); //compute the mean of the squared SNP

		for (size_t iSnp = 0; iSnp < nSNPs; ++iSnp)
		{
			if (n_observed[iSnp] < 1.0)
			{
				printf("No individual observed for the SNP.\n");
			}

			mean_s[iSnp] = sum_s[iSnp] / n_observed[iSnp];    //compute the mean over observed individuals for the current SNP
			mean2_s[iSnp] = sum2_s[iSnp] / n_observed[iSnp];    //compute the mean of the squared SNP

			if ((mean_s[iSnp] != mean_s[iSnp]) || (betaNotUnitVariance && ((mean_s[iSnp] > (REAL)2.0) || (mean_s[iSnp] < (REAL)0.0))))
			{
				if (!seenSNC)
				{
					seenSNC = true;
					fprintf(stderr, "Illegal SNP mean: %.2f for SNPs[:][%zu]\n", mean_s[iSnp], iSnp);
				}
			}


			REAL variance = mean2_s[iSnp] - mean_s[iSnp] * mean_s[iSnp];        //By the Cauchy Shwartz inequality this should always be positive
			std[iSnp] = sqrt(variance);

			if ((std[iSnp] != std[iSnp]) || (std[iSnp] <= (REAL)0.0))
			{
				// a std == 0.0 means all SNPs have the same value (no variation or Single Nucleotide Constant (SNC))
				//   however ALL SNCs should have been removed in previous filtering steps
				//   This test now prevents a divide by zero error below
				std[iSnp] = std::numeric_limits<REAL>::infinity();
				isSNC[iSnp] = true;
				if (!seenSNC)
				{
					seenSNC = true;
					// Don't need this warning because SNCs are still meaning full in QQ plots because they should be thought of as SNPs without enough data.
					// fprintf(stderr, "std=.%2f has illegal value for SNPs[:][%zu]\n", std[iSnp], iSnp);
				}
			}
			stats[iSnp*2] = mean_s[iSnp];
			stats[iSnp*2+1] = std[iSnp];
		}
	}

	if (apply_in_place)
	{
		for (size_t ind = 0; ind < nIndividuals; ++ind)
		{
			size_t rowStart = ind * nSNPs;
			for (size_t iSnp = 0; iSnp < nSNPs; ++iSnp)
			{
				REAL value = SNPs[rowStart + iSnp];
				//check for NaN
				if ((value != value) || isSNC[iSnp])
				{
					value = 0.0;
				}
				else
				{
					value -= mean_s[iSnp];     //subtract the mean from the data
					if (betaNotUnitVariance)
					{
						//compute snp-freq as in the Visscher Height paper (Nat Gen, Yang et al 2010).
						REAL freq = mean_s[iSnp] / 2.0;
						if (freq > .5)
						{
							freq = 1.0 - freq;
						}

						REAL rT = SUFFIX(BetaPdf)(freq, betaA, betaB);
						//fprintf(stderr, "BetaPdf(%f,%f,%f)=%f\n",  freq, betaA, betaB, rT);
						value *= rT;
					}
					else
					{
						value /= std[iSnp];        //unit variance as well
					}
				}
				SNPs[rowStart + iSnp] = value;
			}
		}
	}
#endif
}
