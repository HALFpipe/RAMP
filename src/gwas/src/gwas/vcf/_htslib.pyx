# cython: language_level=3
from cython cimport cast, wraparound
from libc.stdint cimport int64_t, int32_t, uint32_t
from libc.stdlib cimport free
import numpy as np
cimport numpy as np

np.import_array()


# Adapted from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pxd
cdef extern from "htslib/hts.h":
    ctypedef struct htsFile:
        pass

    htsFile *hts_open(char *fn, char *mode)


cdef extern from "htslib/vcf.h":
    cdef extern const int BCF_DT_ID
    cdef extern const int BCF_UN_INFO

    ctypedef union uv1:
        int32_t i
        float f

    ctypedef struct bcf_fmt_t:
        int id
        int n  # n

    ctypedef struct bcf_info_t:
        # The key is the numeric tag id
        # The corresponding string is bcf_hdr_t::id[BCF_DT_ID][$key].key
        int key
        uv1 v1

    ctypedef struct bcf_dec_t:
        # allele[0] is the reference allele
        char **allele
        bcf_info_t *info

    ctypedef struct bcf1_t:
        int64_t pos
        int32_t rid  # Chromosome
        uint32_t n_info, n_allele
        bcf_dec_t d

    ctypedef struct bcf_hdr_t:
        char **samples

    bcf1_t * bcf_init() nogil
    void bcf_destroy(bcf1_t *v) nogil

    bcf_hdr_t *bcf_hdr_read(htsFile *fp) nogil

    int bcf_hdr_nsamples(const bcf_hdr_t *hdr) nogil

    int hts_close(htsFile *fp) nogil

    int bcf_read(htsFile *fp, const bcf_hdr_t *h, bcf1_t *v) nogil

    const char *bcf_hdr_id2name(const bcf_hdr_t *hdr, int rid) nogil
    int bcf_hdr_id2int(const bcf_hdr_t *hdr, int type, const char *id) nogil

    int bcf_unpack(bcf1_t *b, int which) nogil

    bcf_fmt_t *bcf_get_fmt(const bcf_hdr_t *hdr, bcf1_t *line, const char *key) nogil

    int bcf_get_format_float(
        const bcf_hdr_t *hdr,
        bcf1_t *line,
        char* tag,
        float **dst,
        int *ndst
    ) nogil

    void bcf_hdr_destroy(const bcf_hdr_t *hdr) nogil


cdef void open_vcf_file(
    str file_path,
    htsFile** fp,
    bcf_hdr_t** hdr,
    bcf1_t** variant,
):
    path_bytes = file_path.encode()
    cdef char* path_char = path_bytes

    fp[0] = hts_open(path_char, b"r")
    if fp[0] == NULL:
        raise IOError(f"failed to read \"{file_path}\"")

    hdr[0] = bcf_hdr_read(fp[0])
    if hdr[0] == NULL:
        hts_close(fp[0])
        raise RuntimeError(f"failed to read header from \"{file_path}\"")

    variant[0] = bcf_init()
    if variant[0] == NULL:
        bcf_hdr_destroy(hdr[0])
        hts_close(fp[0])
        raise RuntimeError(f"failed to get variant from \"{file_path}\"")


cdef read_variant(bcf_hdr_t* hdr, bcf1_t* record, object variant_parser):
    # Adapted from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pyx#L2361

    # As per https://github.com/samtools/htslib/issues/848
    # BCF_UN_STR  1       // up to ALT inclusive
    # BCF_UN_FLT  2       // up to FILTER
    # BCF_UN_INFO 4       // up to INFO
    # BCF_UN_FMT  8       // unpack format and each sample
    bcf_unpack(record, BCF_UN_INFO)

    cdef int64_t position = record.pos + 1  # Positions are off by one

    cdef int alternate_allele_frequency_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"AF")
    if alternate_allele_frequency_key < 0:
        raise ValueError()
    cdef int minor_allele_frequency_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"MAF")
    if minor_allele_frequency_key < 0:
        raise ValueError()
    cdef int imputed_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"IMPUTED")
    if imputed_key < 0:
        raise ValueError()
    cdef int r2_value_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"R2")
    if r2_value_key < 0:
        raise ValueError()

    cdef float alternate_allele_frequency = np.nan
    cdef float minor_allele_frequency = np.nan
    cdef bint is_imputed = False
    cdef float r2_value = np.nan

    cdef bcf_info_t *info
    for i in range(record.n_info):
        info = &record.d.info[i]
        if info.key == alternate_allele_frequency_key:
            alternate_allele_frequency = info.v1.f
        elif info.key == minor_allele_frequency_key:
            minor_allele_frequency = info.v1.f
        elif info.key == imputed_key:
            is_imputed = True
        elif info.key == r2_value_key:
            r2_value = info.v1.f

    cdef const char* chromosome = bcf_hdr_id2name(hdr, record.rid)
    cdef char* reference_allele = record.d.allele[0]
    cdef char* alternate_allele = record.d.allele[1]

    return variant_parser(
        chromosome.decode(),
        position,
        reference_allele.decode(),
        alternate_allele.decode(),
        is_imputed,
        alternate_allele_frequency,
        minor_allele_frequency,
        r2_value,
    )


def read_variants(str file_path, object variant_parser):
    cdef htsFile* fp
    cdef bcf_hdr_t* hdr
    cdef bcf1_t* variant
    open_vcf_file(file_path, &fp, &hdr, &variant)

    cdef int32_t sample_count = bcf_hdr_nsamples(hdr)

    cdef int status

    cdef list samples = [hdr.samples[i].decode() for i in range(sample_count)]
    cdef list variants = []
    while True:
        status = bcf_read(fp, hdr, variant)

        if status == -1:
            break
        elif status < -1:
            raise IOError(f"Error reading VCF file: {status}")

        variants.append(read_variant(hdr, variant, variant_parser))

    bcf_destroy(variant)
    bcf_hdr_destroy(hdr)
    hts_close(fp)

    return variants, samples


@wraparound(False)
def read_dosages(
    str file_path,
    np.ndarray[np.float64_t, ndim=2] dosages,
    np.ndarray[np.uint32_t, ndim=1] sample_indices,
    np.ndarray[np.uint32_t, ndim=1] variant_indices
):
    cdef htsFile* fp
    cdef bcf_hdr_t* hdr
    cdef bcf1_t* variant
    open_vcf_file(file_path, &fp, &hdr, &variant)

    cdef char* tag = b"DS"

    cdef uint32_t variant_index = 0
    cdef uint32_t variant_indices_index = 0
    cdef uint32_t sample_index
    cdef uint32_t sample_indices_index

    cdef int status
    cdef bcf_fmt_t *fmt
    cdef int count
    cdef float* buffer
    cdef int buffer_size
    cdef np.float64_t value

    while variant_indices_index < variant_indices.size:
        status = bcf_read(fp, hdr, variant)

        if status == -1:
            raise EOFError("Reached end of file before reading all requested variants")
        elif status < -1:
            raise IOError(f"Error reading VCF file: {status}")

        if variant_index == variant_indices[variant_indices_index]:
            fmt = bcf_get_fmt(hdr, variant, tag)
            if fmt.n != 1:
                raise ValueError(f"Expected one value per sample, but got {fmt.n}")

            buffer = NULL
            buffer_size = 0
            count = bcf_get_format_float(hdr, variant, tag, &buffer, &buffer_size)

            if count == -1:
                raise ValueError("No such INFO tag defined in the header")
            elif count == -2:
                raise ValueError(
                    "Clash between types defined in the header "
                    "and encountered in the VCF record"
                )
            elif count == -3:
                raise ValueError("Tag is not present in the VCF record")
            elif count == -4:
                raise RuntimeError(
                    "The operation could not be completed (e.g. out of memory)"
                )
            elif count < 0:
                raise RuntimeError("Unknown error")

            if buffer == NULL:
                raise RuntimeError("Received NULL buffer")

            sample_indices_index = 0
            while sample_indices_index < sample_indices.size:
                sample_index = sample_indices[sample_indices_index]
                if sample_index >= cast(uint32_t, count):
                    raise ValueError(
                        f"Sample index {sample_index} is out of "
                        f"bounds of the buffer size {count}"
                    )
                value = buffer[sample_index]
                dosages[variant_indices_index, sample_indices_index] = value
                sample_indices_index += 1

            free(buffer)

            variant_indices_index += 1
        variant_index += 1

    bcf_destroy(variant)
    bcf_hdr_destroy(hdr)
    hts_close(fp)
