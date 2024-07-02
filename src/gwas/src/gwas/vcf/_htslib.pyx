# cython: language_level=3
from libc.stdint cimport int64_t, uint64_t, int32_t, uint32_t, int8_t, int16_t, uint8_t
import numpy as np
cimport numpy as np
np.import_array()
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc cimport stdlib


# cdef extern from "htslib/hts.h":
#     ctypedef struct htsFile:
#         pass
# 
#     htsFile *hts_open(const char *fn, const char *mode)
#     int hts_close(htsFile *fp)

# from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pxd
cdef extern from "htslib/hfile.h":
    ctypedef struct hFILE:
        pass
    hFILE *hdopen(int fd, const char *mode);

cdef extern from "htslib/hts.h":


    int hts_set_threads(htsFile *fp, int n);


    cdef union ufp:
        hFILE *hfile;

    cdef enum htsExactFormat:
        unknown_format,
        binary_format, text_format,
        sam, bam, bai, cram, crai, vcf, bcf, csi, gzi, tbi, bed

    ctypedef struct htsFormat:
        htsExactFormat format
    
    ctypedef struct htsFile:
        ufp fp
        htsFormat format

    int hts_detect_format(hFILE *fp, htsFormat *fmt);

    htsFile *hts_open(char *fn, char *mode);

    htsFile *hts_hopen(hFILE *fp, const char *fn, const char *mode);

    cdef int hts_verbose = 1

    ctypedef struct hts_itr_t:
        pass

    ctypedef struct hts_idx_t:
        pass

    hts_idx_t *bcf_index_load(char *fn)
    hts_idx_t *hts_idx_load2(const char *fn, const char *fnidx);
    int hts_idx_nseq(const hts_idx_t *idx);
    int hts_idx_get_stat(const hts_idx_t* idx, int tid, uint64_t* mapped,
            uint64_t* unmapped);

    #int hts_itr_next(BGZF *fp, hts_itr_t *iter, void *r, void *data);
    void hts_itr_destroy(hts_itr_t *iter);
    void hts_idx_destroy(hts_idx_t *idx);

cdef extern from "htslib/tbx.h":
    ctypedef struct tbx_conf_t:
        pass

    # Expose details of tbx_t so that we can access the idx field
    ctypedef struct tbx_t:
        tbx_conf_t conf
        hts_idx_t *idx
        void *dict

    tbx_t *tbx_index_load(const char *fn);
    tbx_t *tbx_index_load2(const char *fn, const char *fnidx);
    hts_itr_t *tbx_itr_queryi(tbx_t *tbx, int tid, int64_t beg, int64_t end)
    hts_itr_t *tbx_itr_querys(tbx_t *tbx, char *reg) nogil
    int tbx_itr_next(htsFile *fp, tbx_t *tbx, hts_itr_t *iter, void *data) nogil;
    void tbx_destroy(tbx_t *tbx);


cdef extern from "htslib/vcf.h":

    ctypedef struct hts_idx_t:
        pass

    int bcf_itr_next(htsFile *, hts_itr_t* iter, bcf1_t*)
    hts_itr_t *bcf_itr_querys(hts_idx_t *, void *, char *);


    cdef extern const int BCF_DT_ID;
    cdef extern const int BCF_DT_CTG;
    cdef extern const int BCF_DT_SAMPLE;

    cdef extern uint32_t bcf_float_missing;

    cdef extern const int BCF_ERR_CTG_UNDEF;


    cdef extern const int BCF_BT_NULL;
    cdef extern const int BCF_BT_INT8;
    cdef extern const int BCF_BT_INT16;
    cdef extern const int BCF_BT_INT32;
    cdef extern const int BCF_BT_FLOAT;
    cdef extern const int BCF_BT_CHAR;

    cdef extern const int bcf_str_missing;
    cdef extern const int bcf_str_vector_end;

    cdef extern const int INT8_MIN;
    cdef extern const int INT16_MIN;
    cdef extern const int INT32_MIN;

    cdef extern const int bcf_int8_vector_end;
    cdef extern const int bcf_int16_vector_end;
    cdef extern const int bcf_int32_vector_end;

    cdef extern const int bcf_int8_missing;
    cdef extern const int bcf_int16_missing;
    cdef extern const int32_t bcf_int32_missing;

    ctypedef union uv1:
        int32_t i; # integer value
        float f;   # float value

    ctypedef struct variant_t:
        pass

    ctypedef struct bcf_fmt_t:
        int id;
        int n; # n

    ctypedef struct bcf_info_t:
        int key;        # key: numeric tag id, the corresponding string is bcf_hdr_t::id[BCF_DT_ID][$key].key
        int type;  # type: one of BCF_BT_* types; len: vector length, 1 for scalars
        #} v1; # only set if $len==1; for easier access
        uv1 v1
        uint8_t *vptr;          # pointer to data array in bcf1_t->shared.s, excluding the size+type and tag id bytes
        uint32_t vptr_len;      # length of the vptr block or, when set, of the vptr_mod block, excluding offset
        uint32_t vptr_off;
        uint32_t vptr_free;   # vptr offset, i.e., the size of the INFO key plus size+type bytes
        int len;
               # indicates that vptr-vptr_off must be freed; set only when modified and the new


    ctypedef struct bcf_dec_t:
        int m_fmt, m_info, m_id, m_als, m_allele, m_flt; # allocated size (high-water mark); do not change
        int n_flt;  # Number of FILTER fields
        int *flt;   # FILTER keys in the dictionary
        char *id;      # ID block (\0-seperated)
        char *als;     # REF+ALT block (\0-seperated)
        char **allele;      # allele[0] is the REF (allele[] pointers to the als block); all null terminated
        bcf_info_t *info;   # INFO
        bcf_fmt_t *fmt;     # FORMAT and individual sample
        variant_t *var;     # $var and $var_type set only when set_variant_types called
        int n_var, var_type;
        int shared_dirty;   # if set, shared.s must be recreated on BCF output
        int indiv_dirty;    # if set, indiv.s must be recreated on BCF output

    ctypedef struct bcf1_t:
        int64_t pos;  #// POS
        int64_t rlen; #// length of REF
        int32_t rid;  #// CHROM
        float qual;   #// QUAL
        uint32_t n_info, n_allele;
        uint32_t n_fmt #//:8 #//, n_sample:24;
        #kstring_t shared, indiv;
        bcf_dec_t d; #// lazy evaluation: $d is not generated by bcf_read(), but by explicitly calling bcf_unpack()
        int max_unpack;        # // Set to BCF_UN_STR, BCF_UN_FLT, or BCF_UN_INFO to boost performance of vcf_parse when some of the fields wont be needed
        int unpacked;          # // remember what has been unpacked to allow calling bcf_unpack() repeatedly without redoing the work
        int unpack_size[3];    # // the original block size of ID, REF+ALT and FILTER
        int errcode;   # // one of BCF_ERR_* codes

    ctypedef struct bcf_idpair_t:
        pass

    cdef extern const int BCF_HL_FLT; # header line
    cdef extern const int BCF_HL_INFO;
    cdef extern const int BCF_HL_FMT;
    cdef extern const int BCF_HL_CTG;
    cdef extern const int BCF_HL_STR; # structured header line TAG=<A=..,B=..>
    cdef extern const int BCF_HL_GEN; # generic header line

    cdef extern const int BCF_HT_FLAG; # header type
    cdef extern const int BCF_HT_INT;
    cdef extern const int BCF_HT_REAL;
    cdef extern const int BCF_HT_STR;

    ctypedef struct bcf_hrec_t:
        int type;       # One of the BCF_HL_* type
        char *key;      # The part before '=', i.e. FILTER/INFO/FORMAT/contig/fileformat etc.
        char *value;    # Set only for generic lines, NULL for FILTER/INFO, etc.
        int nkeys;              # Number of structured fields
        char **keys;    # The key=value pairs
        char **vals;    # The key=value pairs

    ctypedef struct kstring_t:
        pass

    ctypedef struct bcf_hdr_t:
        int32_t n[3];
        bcf_idpair_t *id[3];
        void *dict[3];         # ID dictionary, contig dict and sample dict
        char **samples;
        bcf_hrec_t **hrec;
        int nhrec, dirty;
        int ntransl;    # for bcf_translate()
        int *transl[2]; # for bcf_translate()
        int nsamples_ori;        # for bcf_hdr_set_samples()
        uint8_t *keep_samples;
        kstring_t mem;
        int32_t m[3];


    void bcf_float_set(float *ptr, uint32_t value)
    bint bcf_float_is_missing(float f)
    bint bcf_float_is_vector_end(float f)

    void bcf_destroy(bcf1_t *v);
    bcf1_t * bcf_init() nogil;
    int vcf_parse(kstring_t *s, const bcf_hdr_t *h, bcf1_t *v) nogil;
    int bcf_subset_format(const bcf_hdr_t *hdr, bcf1_t *rec);

    int bcf_update_alleles(const bcf_hdr_t *hdr, bcf1_t *line, const char **alleles, int nals);
    int bcf_update_alleles_str(const bcf_hdr_t *hdr, bcf1_t *line, const char *alleles_string);

    bcf_hdr_t *bcf_hdr_read(htsFile *fp);

    int bcf_hdr_set_samples(bcf_hdr_t *hdr, const char *samples, int is_file);
    int bcf_hdr_nsamples(const bcf_hdr_t *hdr);
    void bcf_hdr_destroy(const bcf_hdr_t *hdr)
    char *bcf_hdr_fmt_text(const bcf_hdr_t *hdr, int is_bcf, int *len);
    int bcf_hdr_format(const bcf_hdr_t *hdr, int is_bcf, kstring_t *str);

    bcf_hdr_t *bcf_hdr_init(const char *mode);
    int bcf_hdr_parse(bcf_hdr_t *hdr, char *htxt);

    int bcf_write(htsFile *fp, const bcf_hdr_t *h, bcf1_t *v);
    int bcf_hdr_write(htsFile *fp, bcf_hdr_t *h);
    int vcf_format(const bcf_hdr_t *h, const bcf1_t *v, kstring_t *s);

    bcf_hrec_t *bcf_hdr_get_hrec(const bcf_hdr_t *hdr, int type, const char *key, const char *value, const char *str_class);
    void bcf_hrec_destroy(bcf_hrec_t *)
    bcf_hrec_t *bcf_hdr_id2hrec(const bcf_hdr_t *hdr, int type, int idx, int rid);
    int bcf_hdr_add_hrec(bcf_hdr_t *hdr, bcf_hrec_t *hrec);

    int hts_close(htsFile *fp);

    int bcf_read(htsFile *fp, const bcf_hdr_t *h, bcf1_t *v) nogil;

    const char *bcf_hdr_id2name(const bcf_hdr_t *hdr, int rid);
    const char *bcf_hdr_int2id(const bcf_hdr_t *hdr, int type, int int_id)
    int bcf_hdr_id2int(const bcf_hdr_t *hdr, int type, const char *id);
    int bcf_hdr_id2type(bcf_hdr_t * hdr,int type, int int_id)

    int bcf_unpack(bcf1_t *b, int which) nogil;


    bcf_fmt_t *bcf_get_fmt(const bcf_hdr_t *hdr, bcf1_t *line, const char *key);

    int bcf_get_genotypes(const bcf_hdr_t *hdr, bcf1_t *line, int32_t **dst, int *ndst);
    int bcf_get_format_int32(const bcf_hdr_t *hdr, bcf1_t *line, char * tag, int **dst, int *ndst);
    int bcf_get_format_float(const bcf_hdr_t *hdr, bcf1_t *line, char * tag, float **dst, int *ndst)
    int bcf_get_format_string(const bcf_hdr_t *hdr, bcf1_t *line, const char *tag, char ***dst, int *ndst);

    int bcf_get_format_values(const bcf_hdr_t *hdr, bcf1_t *line, const char *tag, void **dst, int *ndst, int type);
    int bcf_gt_is_phased(int);
    int bcf_gt_is_missing(int);
    int bcf_gt_allele(int);
    bint bcf_float_is_missing(float);
    bcf_info_t *bcf_get_info(const bcf_hdr_t *hdr, bcf1_t *line, const char *key);

    int bcf_update_info(const bcf_hdr_t *hdr, bcf1_t *line, const char *key, const void *values, int n, int type);
    int bcf_update_filter(const bcf_hdr_t *hdr, bcf1_t *line, int *flt_ids, int n);




    ## genotypes
    void bcf_gt2alleles(int igt, int *a, int *b);
    int bcf_update_genotypes(const bcf_hdr_t *hdr, bcf1_t *line, const void *values, int n);
    # idx is 0 for ref, 1... for alts...
    int bcf_gt_phased(int idx);
    int bcf_gt_unphased(int idx);

    # sample/format fields
    int bcf_update_format(const bcf_hdr_t *hdr, bcf1_t *line, const char *key, const void *values, int n, int type);
    int bcf_update_format_int32(const bcf_hdr_t * hdr, bcf1_t * line, const char * key, const void * values, int n)
    int bcf_update_format_float(const bcf_hdr_t * hdr, bcf1_t * line, const char * key, const void * values, int n)
    int bcf_update_format_string(const bcf_hdr_t *hdr, bcf1_t *line, const char *key, const char **values, int n);
    int bcf_update_format_char(const bcf_hdr_t *hdr, bcf1_t *line, const char *key, const char**values, int n);

    int bcf_add_id(const bcf_hdr_t *hdr, bcf1_t *line, const char *id);
    int bcf_update_id(const bcf_hdr_t *hdr, bcf1_t *line, const char *id);
    int bcf_hdr_append(bcf_hdr_t * hdr, char *);
    int bcf_hdr_sync(bcf_hdr_t *h);
    bcf_hdr_t *bcf_hdr_dup(bcf_hdr_t *h);

    int bcf_update_info_int32(const bcf_hdr_t *hdr, bcf1_t * line, const char *key, const int32_t *values, int n)
    int bcf_update_info_float(const bcf_hdr_t *hdr, bcf1_t * line, const char *key, const float *values, int n)
    int bcf_update_info_flag(const bcf_hdr_t *hdr, bcf1_t * line, const char
            *key, const char *value, int n)
    int bcf_update_info_string(const bcf_hdr_t *hdr, bcf1_t * line, const char *key, const char *values)
    #define bcf_update_info_flag(hdr,line,key,string,n)    bcf_update_info((hdr),(line),(key),(string),(n),BCF_HT_FLAG)
    #define bcf_update_info_float(hdr,line,key,values,n)   bcf_update_info((hdr),(line),(key),(values),(n),BCF_HT_REAL)
    #define bcf_update_info_flag(hdr,line,key,string,n)    bcf_update_info((hdr),(line),(key),(string),(n),BCF_HT_FLAG)
    #define bcf_update_info_string(hdr,line,key,string)    bcf_update_info((hdr),(line),(key),(string),1,BCF_HT_STR)

    # free the array, not the values.
    char **bcf_index_seqnames(hts_idx_t *idx, bcf_hdr_t *hdr, int *n);
    char **tbx_seqnames(tbx_t *tbx, int *n)
    char **bcf_hdr_seqnames(bcf_hdr_t *hdr, int *n);

# from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pyx#L2361


def read_vcf_file(str path):
    path_bytes = path.encode("utf-8")
    cdef char *path_char = path_bytes
    cdef htsFile *hts = hts_open(path_char, "r")

    # print(hts_close(hts))
    # if not hts:
    #    raise RuntimeError(f"Could not open VCF file at:{path}")
    # return hts

cdef read_variant(bcf_hdr_t* hdr, bcf1_t* record):
    # from https://github.com/samtools/htslib/issues/848
    # BCF_UN_STR  1       // up to ALT inclusive 
    # BCF_UN_FLT  2       // up to FILTER 
    # BCF_UN_INFO 4       // up to INFO 
    # BCF_UN_FMT  8       // unpack format and each sample 
    bcf_unpack(record, 4)

    cdef int64_t pos = record.pos + 1 # positions are off by one
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
    cdef int r2_value_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"R2")
    if r2_value_key < 0:
        r2_value_key = bcf_hdr_id2int(hdr, BCF_DT_ID, b"ER2")
    if r2_value_key < 0:
        raise ValueError()

    cdef float allele_frequency
    cdef float minor_allele_frequency 
    cdef bint imputed
    cdef float r2_value 
    # variant.d.info[minor_allele_frequency_index].v1.f

    cdef bcf_info_t *info
    for i in range(record.n_info):
        info = &record.d.info[i]
        if info.key == allele_frequency_key:
            allele_frequency = info.v1.f
        elif info.key == minor_allele_frequency_key:
            minor_allele_frequency = info.v1.f
        elif info.key == imputed_key:
            imputed = (info.v1.i == 200)
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
    # cdef htsFile* fp = read_vcf_file(file_path)
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

    cdef list variants = []
        
    while bcf_read(fp, hdr, variant) >=0:
        variants.append(read_variant(hdr, variant))

    bcf_destroy(variant)
    bcf_hdr_destroy(hdr)
    hts_close(fp)

    return variants

    # # FORMAT field
    # cdef char** format
    # cdef int i
    # cdef const char* key
    # cdef bcf_fmt_t fmt
    # cdef str format_str

    # from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pyx#L1368
    # keys = []
    # format_str = ""
    # for i in range(variant.n_fmt):
    #     fmt = variant.d.fmt[i]
    #     if fmt.id != -1:
    #         key = bcf_hdr_int2id(hdr, BCF_DT_ID, fmt.id)
    #         if key:
    #             if format_str:
    #                  format_str += ":"
    #             format_str += key.decode("utf-8")
    
        # b":".join([format[i] for i in range(len(format))]).decode() # adapted from https://github.com/brentp/cyvcf2/blob/main/cyvcf2/cyvcf2.pyx property format
        # format_str,


def read(str file_path, np.ndarray dosages, np.ndarray sample_indices):
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
    
    # cdef int32_t n_samples = bcf_hdr_nsamples(hdr)

    cdef int pos_in_dosage = 0
    cdef int n_variants = dosages.shape[0]
        
    while bcf_read(fp, hdr, variant) >=0:
        if pos_in_dosage >= n_variants:
            break
        read_dosages(hdr, variant, dosages, pos_in_dosage, sample_indices)
        pos_in_dosage += 1

    bcf_destroy(variant)
    bcf_hdr_destroy(hdr)
    hts_close(fp)

cdef void read_dosages(bcf_hdr_t* hdr, bcf1_t* record, np.ndarray dosages, int pos_in_dosage, np.ndarray sample_indices):
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
    # cdef bytes tag = to_bytes(field)
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
    elif vtype == "String" or vtype == str or vtype == "Character":
        vtype = str
        nret = bcf_get_format_string(hdr, b, tag, <char ***>&buf, &n)
        typenum = np.NPY_STRING
    else:
        raise Exception("type %s not supported in format()" % vtype)

    if nret < 0:
        return None

    cdef np.npy_intp shape[2]
    cdef char **dst
    cdef int i
    cdef list v
    shape[0] = bcf_hdr_nsamples(hdr)  # number of samples
    shape[1] = fmt.n  # values per sample

    if vtype == str:
        dst = <char **>buf
        v = []
        for i in range(bcf_hdr_nsamples(hdr)):
            v.append(dst[i].decode('utf-8'))
        #v = [dst[i] for i in range(bcf_hdr_nsamples(hdr))]
        xret = np.array(v, dtype=str)
        stdlib.free(dst[0])
        # stdlib.free(dst) # couldnt free because of error can't convert char ** to python object
        return xret

    iv = np.PyArray_SimpleNewFromData(2, shape, typenum, buf)
    array = np.asarray(iv)
    return array