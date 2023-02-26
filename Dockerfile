FROM ubuntu as base

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
    conda config --set channel_priority "strict" && \
    sync && \
    mamba clean --yes --all --force-pkgs-dirs

# Build packages
# ==============
FROM conda as builder

RUN apt-get update && \
    apt-get install --yes --no-install-recommends \
        "git" \
        "subversion" \
        "build-essential"  # need system `ar` for raremetal
RUN mamba install --yes "boa" "conda-verify"

FROM builder as r-gmmat
COPY recipes/r-gmmat r-gmmat
RUN conda mambabuild --no-anaconda-upload "r-gmmat" && \
    conda build purge

FROM builder as r-saige
COPY recipes/r-saige r-saige
RUN conda mambabuild --no-anaconda-upload "r-saige" && \
    conda build purge

FROM builder as qctool
COPY recipes/qctool qctool
RUN conda mambabuild --no-anaconda-upload "qctool" && \
    conda build purge

FROM builder as bolt-lmm
COPY recipes/bolt-lmm bolt-lmm
RUN conda mambabuild --no-anaconda-upload "bolt-lmm" && \
    conda build purge

FROM builder as dosage-convertor
COPY recipes/dosage-convertor dosage-convertor
RUN conda mambabuild --no-anaconda-upload "dosage-convertor" && \
    conda build purge

FROM builder as raremetal
COPY recipes/raremetal raremetal
RUN conda mambabuild --no-anaconda-upload "raremetal" && \
    conda build purge

FROM builder as merge
COPY --from=r-gmmat /usr/local/mambaforge/conda-bld /usr/local/mambaforge/conda-bld
COPY --from=r-saige /usr/local/mambaforge/conda-bld /usr/local/mambaforge/conda-bld
COPY --from=qctool /usr/local/mambaforge/conda-bld /usr/local/mambaforge/conda-bld
COPY --from=bolt-lmm /usr/local/mambaforge/conda-bld /usr/local/mambaforge/conda-bld
COPY --from=dosage-convertor /usr/local/mambaforge/conda-bld /usr/local/mambaforge/conda-bld
COPY --from=raremetal /usr/local/mambaforge/conda-bld /usr/local/mambaforge/conda-bld
RUN conda index /usr/local/mambaforge/conda-bld

# Install packages
# ================
FROM conda as install

COPY --from=merge /usr/local/mambaforge/conda-bld /usr/local/mambaforge/conda-bld
RUN mamba install --yes --use-local \
        "bcftools" \
        "bolt-lmm" \
        "bzip2" \
        "dosage-convertor" \
        "gcta" \
        "gemma" \
        "lrzip" \
        "matplotlib" \
        "parallel" \
        "plink" \
        "plink2" \
        "python>=3.11" \
        "qctool" \
        "raremetal" \
        "r-gmmat" \
        "r-saige" \
        "r-skat" \
        "tabix" && \
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
