/*#if !defined( CPlinkBedFileT_h )
#define CPlinkBedFileT_h
*/
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
 * CPlinkBedFile - {PLINK Bed File Access Class}
 *
 *         File Name:   CPlinkBedFile.h
 *           Version:   2.00
 *            Author:   
 *     Creation Date:    4 Dec 2011
 *     Revision Date:   14 Aug 2013
 *
 *    Module Purpose:   This file defines the CPlinkBedFile class 
 *
 *                      A .BED file contains compressed binary genotype values for 
 *                         for individuals by SNPs.  In other contexts, we prefer and may
 *                         require the file LayoutMode be LayoutGroupGenotypesBySnp
 *
 *                      The .bed header is three bytes followed immediately by data.
 *                      
 *                         bedFileMagic1 | bedFileMagic2 | LayoutMode
 *                         [... data ... ]
 *
 *    Change History:   Version 2.00: Reworked to be wrapped in python version by Chris Widmer (chris@shogun-toolbox.org)
 *
 * Test Files: 
 */

#include <vector>
#include <string>
#include <limits>
 
using namespace std;
typedef unsigned char BYTE;
typedef unsigned long long uint64_t_;


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
   );

