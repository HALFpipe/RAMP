# cython: language_level=3

# Adapted from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pxd
cdef extern from "htslib/hts.h":
    ctypedef struct htsFile:
        pass

    htsFile *hts_open(char *fn, char *mode)
    int hts_close(htsFile *fp)


def read_vcf_file(str path):
    path_bytes = path.encode("utf-8")
    cdef char *path_char = path_bytes
    cdef htsFile *hts = hts_open(path_char, "r")

    print(hts_close(hts))
