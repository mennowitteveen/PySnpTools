#include "MatrixSubsetT.h"
#include <iostream>
#include <stdio.h>
#include <math.h> 
#include <stdlib.h>
#include <stdexcept> //!!!cmk0

using namespace std;

void SUFFIX(matrixSubset)(REALIN* in_, int in_iid_count, int in_sid_count, int val_count, std::vector<size_t> iid_index, std::vector<int> sid_index, REALOUT* out)
{
	uint64_t_ out_iid_count = iid_index.size();
	uint64_t_ out_sid_count = sid_index.size();

	//if (val_count != 1) {//!!!cmk0
	//	std::cout << "val_count != 1 (" << in_iid_count << ", " << in_sid_count << ", " << val_count << ")" << endl;
	//	//throw std::invalid_argument("val_count != 1");
	//}

#ifdef ORDERFIN

	//f-order in "most rapidly changing variable first"
	for (size_t val_index = 0; val_index != val_count; val_index++) {
		//if (val_index != 0) {//!!!cmk0
		//	std::cout << "val_index != 0\n";
		//	//throw std::invalid_argument("val_index != 0");
		//}
		for (size_t sid_index_out = 0; sid_index_out != out_sid_count; sid_index_out++) {
			int sid_index_in = sid_index[sid_index_out];
			REALIN* in2 = in_ + in_iid_count * (uint64_t_)sid_index_in + in_iid_count * in_sid_count * (uint64_t_)val_index;
			//REALIN* in2x = in_ + in_iid_count * (uint64_t_)sid_index_in;
			//if (in2 != in2x) {//!!!cmk0
			//	std::cout << "in2 != in2x\n";
			//	//throw std::invalid_argument("in2 != in2x");
			//}

#ifdef ORDERFOUT //fin,fout
			REALOUT* out2 = out + out_iid_count * (uint64_t_)sid_index_out + out_iid_count * out_sid_count * val_index;
#else            //fin,cout
			REALOUT* out2 = out + sid_index_out * val_count + val_index;
			//REALOUT* out2x = out + sid_index_out;
			//if (out2 != out2x) {//!!!cmk0
			//	std::cout << out2 << " != " << out2x << "(" << out << "," << sid_index_out << "," << val_count << "," << val_index << ")" << endl;
			//	//throw std::invalid_argument("out2 != out2x");
			//}

#endif
			for (size_t iid_index_out = 0; iid_index_out != out_iid_count; iid_index_out++) {
				size_t iid_index_in = iid_index[iid_index_out];

#ifdef ORDERFOUT //fin,fout
				out2[iid_index_out] = (REALOUT)in2[iid_index_in];
#else            //fin,cout
				//if (out_sid_count * val_count * (uint64_t_)iid_index_out != out_sid_count * (uint64_t_)iid_index_out) {//!!!cmk0
				//	std::cout << "not out...\n";
				//	throw std::invalid_argument("not out_sid_count * val_count * (uint64_t_)iid_index_out == out_sid_count * (uint64_t_)iid_index_out");
				//}
				out2[out_sid_count * val_count * (uint64_t_)iid_index_out] = (REALOUT)in2[iid_index_in];
				//!!!cmk0out2[out_sid_count * (uint64_t_)iid_index_out] = (REALOUT)in2[iid_index_in];
#endif
			}
		}
	}

#else
	//c-order in "most rapidly changing index is last"
	for (size_t iid_index_out = 0; iid_index_out != out_iid_count; iid_index_out++) {
		size_t iid_index_in = iid_index[iid_index_out];

		REALIN* in2 = in_ + in_sid_count * val_count * (uint64_t_)iid_index_in;

#ifdef ORDERFOUT //cin,fout
		REALOUT* out2 = out + iid_index_out;
#else            //cin,cout
		REALOUT* out2 = out + out_sid_count * val_count * (uint64_t_)iid_index_out;
#endif

		for (size_t sid_index_out = 0; sid_index_out != out_sid_count; sid_index_out++) {
			int sid_index_in = sid_index[sid_index_out];
			REALIN* in3 = in2 + val_count * (uint64_t_)sid_index_in;
			for (size_t val_index = 0; val_index != val_count; val_index++) {
#ifdef ORDERFOUT //cin,fout
				out2[out_iid_count * (uint64_t_)sid_index_out + out_iid_count * out_sid_count * val_index] = (REALOUT)in3[val_index];
#else            //cin,cout
				out2[sid_index_out * val_count + val_index] = (REALOUT)in3[val_index];
#endif
			}
		}
	}
#endif
}