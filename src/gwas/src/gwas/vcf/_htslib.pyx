# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION
from libc.stdint cimport int64_t, int32_t, uint32_t
import numpy as np
cimport numpy as np
np.import_array()


# from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pxd
cdef extern from "htslib/hfile.h":
    ctypedef struct hFILE:
        pass
    hFILE *hdopen(int fd, const char *mode)

cdef extern from "htslib/hts.h":
    cdef union ufp:
        hFILE *hfile

    cdef enum htsExactFormat:
        unknown_format,
        binary_format, text_format,
        sam, bam, bai, cram, crai, vcf, bcf, csi, gzi, tbi, bed

    ctypedef struct htsFormat:
        htsExactFormat format

    ctypedef struct htsFile:
        ufp fp
        htsFormat format

    htsFile *hts_open(char *fn, char *mode)

    htsFile *hts_hopen(hFILE *fp, const char *fn, const char *mode)


cdef extern from "htslib/vcf.h":
    cdef extern const int BCF_DT_ID

    ctypedef union uv1:
        int32_t i  # integer value
        float f   # float value

    ctypedef struct variant_t:
        pass

    ctypedef struct bcf_fmt_t:
        int id
        int n  # n

    ctypedef struct bcf_info_t:
        int key        # key: numeric tag id,
        # the corresponding string is bcf_hdr_t::id[BCF_DT_ID][$key].key
        int type  # type: one of BCF_BT_* types; len: vector length, 1 for scalars
        # v1; # only set if $len==1; for easier access
        uv1 v1

    ctypedef struct bcf_dec_t:
        char **allele      # allele[0] is the REF (allele[] pointers to the als block)
        bcf_info_t *info   # INFO

    ctypedef struct bcf1_t:
        int64_t pos  # POS
        int32_t rid  # CHROM
        uint32_t n_info, n_allele
        bcf_dec_t d

    ctypedef struct bcf_idpair_t:
        pass

    ctypedef struct kstring_t:
        pass

    ctypedef struct bcf_hdr_t:
        char **samples

    void bcf_destroy(bcf1_t *v)
    bcf1_t * bcf_init() nogil

    bcf_hdr_t *bcf_hdr_read(htsFile *fp)

    int bcf_hdr_nsamples(const bcf_hdr_t *hdr)
    void bcf_hdr_destroy(const bcf_hdr_t *hdr)

    int hts_close(htsFile *fp)

    int bcf_read(htsFile *fp, const bcf_hdr_t *h, bcf1_t *v) nogil

    const char *bcf_hdr_id2name(const bcf_hdr_t *hdr, int rid)
    int bcf_hdr_id2int(const bcf_hdr_t *hdr, int type, const char *id)

    int bcf_unpack(bcf1_t *b, int which) nogil

    bcf_fmt_t *bcf_get_fmt(const bcf_hdr_t *hdr, bcf1_t *line, const char *key)

    int bcf_get_format_int32(
        const bcf_hdr_t *hdr,
        bcf1_t *line,
        char * tag, int **dst,
        int *ndst
        )
    int bcf_get_format_float(
        const bcf_hdr_t *hdr,
        bcf1_t *line, char * tag,
        float **dst, int *ndst
        )

# from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pyx#L2361

cdef read_variant(bcf_hdr_t* hdr, bcf1_t* record):
    # from https://github.com/samtools/htslib/issues/848
    # BCF_UN_STR  1       // up to ALT inclusive
    # BCF_UN_FLT  2       // up to FILTER
    # BCF_UN_INFO 4       // up to INFO
    # BCF_UN_FMT  8       // unpack format and each sample
    bcf_unpack(record, 4)

    cdef int64_t pos = record.pos + 1  # positions are off by one
    cdef const char* chrom = bcf_hdr_id2name(hdr, record.rid)
    cdef char* ref = record.d.allele[0]
    cdef char* alt = record.d.allele[1]

    cdef int allele_frequency_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"AF")
    if allele_frequency_key < 0:
        raise ValueError()
    cdef int minor_allele_frequency_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"MAF")
    if minor_allele_frequency_key < 0:
        raise ValueError()
    cdef int imputed_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"IMPUTED")
    if imputed_key < 0:
        raise ValueError()
    cdef int typed_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"TYPED")
    if typed_key < 0:
        raise ValueError()
    cdef int typed_only_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"TYPED_ONLY")
    if typed_only_key < 0:
        raise ValueError()
    cdef int r2_value_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"R2")

    if r2_value_key < 0:
        raise ValueError()

    cdef float allele_frequency = np.nan
    cdef float minor_allele_frequency = np.nan
    cdef bint imputed = False
    cdef float r2_value = np.nan

    cdef bcf_info_t *info
    for i in range(record.n_info):
        info = &record.d.info[i]
        if info.key == allele_frequency_key:
            allele_frequency = info.v1.f
        elif info.key == minor_allele_frequency_key:
            minor_allele_frequency = info.v1.f
        elif info.key == imputed_key:
            imputed = True  # mere presence of key suffices for imputed flag
        elif info.key == typed_key:
            imputed = False  # typed means both genotyped and imputed
        elif info.key == typed_only_key:
            imputed = False
        elif info.key == r2_value_key:
            r2_value = info.v1.f

    return (
        chrom.decode(),
        pos,
        ref.decode(),
        alt.decode(),
        imputed,
        allele_frequency,
        minor_allele_frequency,
        r2_value,
    )


def read_vcf_records(str file_path):
    path_bytes = file_path.encode("utf-8")
    cdef char* path_char = path_bytes
    cdef htsFile* fp = hts_open(path_char, b"r")
    if not fp:
        raise RuntimeError(f"failed to read {file_path}")

    cdef bcf_hdr_t* hdr = bcf_hdr_read(fp)
    if not hdr:
        hts_close(fp)
        raise RuntimeError(f"failed to read header from {file_path}")

    cdef bcf1_t* variant = bcf_init()
    if not variant:
        bcf_hdr_destroy(hdr)
        hts_close(fp)
        raise RuntimeError(f"failed to read variant from {file_path}")

    cdef int32_t n_samples = bcf_hdr_nsamples(hdr)
    cdef list samples = [hdr.samples[i].decode("utf-8") for i in range(n_samples)]

    cdef list variants = []

    while bcf_read(fp, hdr, variant) >=0:
        variants.append(read_variant(hdr, variant))

    bcf_destroy(variant)
    bcf_hdr_destroy(hdr)
    hts_close(fp)

    return variants, samples


def read(
    str file_path,
    np.ndarray dosages,
    np.ndarray sample_indices,
    np.ndarray variant_indices
        ):
    path_bytes = file_path.encode("utf-8")
    cdef char* path_char = path_bytes
    cdef htsFile* fp = hts_open(path_char, b"r")
    if not fp:
        raise RuntimeError(f"failed to read {file_path}")

    cdef bcf_hdr_t* hdr = bcf_hdr_read(fp)
    if not hdr:
        hts_close(fp)
        raise RuntimeError(f"failed to read header from {file_path}")

    cdef bcf1_t* variant = bcf_init()
    if not variant:
        bcf_hdr_destroy(hdr)
        hts_close(fp)
        raise RuntimeError(f"failed to read variant from {file_path}")

    cdef int pos_in_dosage = 0
    cdef int n_variants = dosages.shape[0]

    cdef int variant_counter = 0
    cdef int var_indices_counter = 0
    cdef int n_variant_indices = variant_indices.shape[0]

    while bcf_read(fp, hdr, variant) >=0:
        if var_indices_counter >= n_variant_indices:
            break
        if variant_counter == variant_indices[var_indices_counter]:
            if pos_in_dosage >= n_variants:
                break
            read_dosages(hdr, variant, dosages, pos_in_dosage, sample_indices)
            pos_in_dosage += 1
            var_indices_counter += 1
        variant_counter += 1

    bcf_destroy(variant)
    bcf_hdr_destroy(hdr)
    hts_close(fp)


cdef void read_dosages(
    bcf_hdr_t* hdr,
    bcf1_t* record,
    np.ndarray dosages,
    int pos_in_dosage,
    np.ndarray sample_indices
        ):
    cdef np.ndarray dosage_fields = format_field(hdr, record, b"DS", "Float")
    if dosage_fields is None:
        dosages[pos_in_dosage, :] = np.nan
    else:
        dosage_fields = np.asarray(dosage_fields)
        dosage_fields = dosage_fields[sample_indices]
        if dosage_fields.ndim > 1:
            dosage_fields = dosage_fields.ravel()
        dosages[pos_in_dosage, :] = dosage_fields


cdef np.ndarray format_field(bcf_hdr_t* hdr, bcf1_t* b, char* field, object vtype):
    cdef bytes tag = field
    cdef bcf_fmt_t *fmt = bcf_get_fmt(hdr, b, tag)
    cdef int n = 0, nret
    cdef void *buf = NULL
    cdef int typenum = 0

    if vtype == "Integer" or vtype == int:
        nret = bcf_get_format_int32(hdr, b, tag, <int **>&buf, &n)
        typenum = np.NPY_INT32
    elif vtype == "Float" or vtype == float:
        nret = bcf_get_format_float(hdr, b, tag, <float **>&buf, &n)
        typenum = np.NPY_FLOAT32
    else:
        raise Exception("type %s not supported in format()" % vtype)

    if nret < 0:
        return None

    cdef np.npy_intp shape[2]

    shape[0] = bcf_hdr_nsamples(hdr)  # number of samples
    shape[1] = fmt.n  # values per sample

    iv = np.PyArray_SimpleNewFromData(2, shape, typenum, buf)
    array = np.asarray(iv)
    return array
