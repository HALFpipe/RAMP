FROM ubuntu:rolling as base

ENV LC_ALL="C.UTF-8" \
    LANG="C.UTF-8" \
    DEBIAN_FRONTEND="noninteractive" \
    DEBCONF_NOWARNINGS="yes" \
    PATH="/usr/local/mambaforge/bin:$PATH"

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
    "ca-certificates" \
    "curl" \
    "gdb" \
    "git" "git-lfs" \
    "libc6-dbg" \
    "libencode-perl" \
    "libfindbin-libs-perl" \
    "less" \
    "valgrind" && \
    apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Install conda
# =============
FROM base as conda

RUN curl --silent --show-error --location \
    "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh" \
    --output "conda.sh" &&  \
    bash conda.sh -b -p /usr/local/mambaforge && \
    rm conda.sh && \
    conda config --system --append channels "bioconda" && \
    conda config --system --prepend channels "conda-forge/label/cython_dev" && \
    sync && \
    mamba clean --yes --all --force-pkgs-dirs

# Build packages
# ==============
FROM conda as builder

RUN mamba install --yes "boa" "conda-verify"
COPY recipes/conda_build_config.yaml /root/conda_build_config.yaml

COPY recipes/bolt-lmm bolt-lmm
RUN conda mambabuild --no-anaconda-upload "bolt-lmm" && \
    conda build purge

COPY recipes/dosage-convertor dosage-convertor
RUN conda mambabuild --no-anaconda-upload "dosage-convertor" && \
    conda build purge

COPY recipes/gcta gcta
RUN conda mambabuild --no-anaconda-upload "gcta" && \
    conda build purge

COPY recipes/qctool qctool
RUN conda mambabuild --no-anaconda-upload "qctool" && \
    conda build purge

COPY recipes/raremetal raremetal
RUN conda mambabuild --no-anaconda-upload "raremetal" && \
    conda build purge

COPY recipes/r-gmmat r-gmmat
RUN conda mambabuild --no-anaconda-upload "r-gmmat" && \
    conda build purge

COPY recipes/r-saige r-saige
RUN conda mambabuild --no-anaconda-upload "r-saige" && \
    conda build purge

COPY recipes/python-blosc2 python-blosc2
RUN conda mambabuild --no-anaconda-upload "python-blosc2" && \
    conda build purge

COPY src/gwas gwas-protocol/src/gwas
COPY recipes/gwas gwas-protocol/recipes/gwas
# copy .git folder too for setuptools_scm
COPY .git gwas-protocol/.git
RUN cd gwas-protocol/recipes && \
    conda mambabuild --no-anaconda-upload "gwas" && \
    conda build purge

RUN conda index /usr/local/mambaforge/conda-bld

# Install packages
# ================
FROM conda as install

COPY --from=builder /usr/local/mambaforge/conda-bld /usr/local/mambaforge/conda-bld
RUN mamba install --yes --use-local \
    "python>=3.11" \
    "pytorch=*=cpu*" \
        "bcftools>=1.17" \
    "gemma" \
        "plink" \
        "plink2" \
        "r-skat" \
        "tabix" \
    "bzip2" \
    "p7zip>=15.09" \
    "parallel" \
        "bolt-lmm" \
        "dosage-convertor" \
        "gcta" \
        "gwas" \
        "python-blosc2" \
        "qctool" \
        "raremetal" \
        "r-gmmat" \
        "r-saige" && \
    sync && \
    rm -rf /usr/local/mambaforge/conda-bld && \
    mamba clean --yes --all --force-pkgs-dirs && \
    sync && \
    find /usr/local/mambaforge/ -follow -type f -name "*.a" -delete && \
    sync


# Final
# =====
FROM base
COPY --from=install /usr/local/mambaforge /usr/local/mambaforge
