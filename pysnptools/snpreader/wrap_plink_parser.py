from bed_reader import wrap_plink_parser_onep

# To help other programs with backwards compatibility

def readPlinkBedFile2doubleFAAA(bed_fn, input_num_ind, input_num_snps, count_A1, iidIdxList, snpIdxList, out):
    wrap_plink_parser_onep.readPlinkBedFile2doubleFAAA(bed_fn, input_num_ind, input_num_snps, count_A1, iidIdxList, snpIdxList, out, 1)

def readPlinkBedFile2doubleCAAA(bed_fn, input_num_ind, input_num_snps, count_A1, iidIdxList, snpIdxList, out):
    wrap_plink_parser_onep.readPlinkBedFile2doubleCAAA(bed_fn, input_num_ind, input_num_snps, count_A1, iidIdxList, snpIdxList, out, 1)

def readPlinkBedFile2floatFAAA(bed_fn, input_num_ind, input_num_snps, count_A1, iidIdxList, snpIdxList, out):
    wrap_plink_parser_onep.readPlinkBedFile2floatFAAA(bed_fn, input_num_ind, input_num_snps, count_A1, iidIdxList, snpIdxList, out, 1)

def readPlinkBedFile2floatCAAA(bed_fn, input_num_ind, input_num_snps, count_A1, iidIdxList, snpIdxList, out):
    wrap_plink_parser_onep.readPlinkBedFile2floatCAAA(bed_fn, input_num_ind, input_num_snps, count_A1, iidIdxList, snpIdxList, out, 1)
