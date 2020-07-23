#define REAL double
#define ORDERC
#undef ORDERF
#define SUFFIX(NAME) NAME ## doubleCAAA
REAL SUFFIX(unknownOrMissing) = std::numeric_limits<REAL>::quiet_NaN();  // now used by SnpInfo
#include "CPlinkBedFileT.h"

#define REAL float
#define ORDERC
#undef ORDERF
#define SUFFIX(NAME) NAME ## floatCAAA
REAL SUFFIX(unknownOrMissing) = std::numeric_limits<REAL>::quiet_NaN();  // now used by SnpInfo
#include "CPlinkBedFileT.h"

#define REAL double
#define ORDERF
#undef ORDERC
#define SUFFIX(NAME) NAME ## doubleFAAA
REAL SUFFIX(unknownOrMissing) = std::numeric_limits<REAL>::quiet_NaN();  // now used by SnpInfo
#include "CPlinkBedFileT.h"

#define REAL float
#define ORDERF
#undef ORDERC
#define SUFFIX(NAME) NAME ## floatFAAA
REAL SUFFIX(unknownOrMissing) = std::numeric_limits<REAL>::quiet_NaN();  // now used by SnpInfo
#include "CPlinkBedFileT.h"

#define REAL signed char
#define ORDERF
#undef ORDERC
#define SUFFIX(NAME) NAME ## int8FAAA
REAL SUFFIX(unknownOrMissing) = -127;  // now used by SnpInfo
#include "CPlinkBedFileT.h"

#define REAL signed char
#define ORDERC
#undef ORDERF
#define SUFFIX(NAME) NAME ## int8CAAA
REAL SUFFIX(unknownOrMissing) = -127;  // now used by SnpInfo
#include "CPlinkBedFileT.h"
